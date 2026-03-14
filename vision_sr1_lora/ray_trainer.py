"""
Vision-SR1 Self-Reward Trainer with Multi-Advantage GRPO.

Extends EasyR1's RayPPOTrainer with:
1. Second-hop generation: after the first rollout (with image), extract the
   <description> and use it to answer the question text-only.
2. Multi-advantage GRPO: compute GRPO advantages for each reward component
   (accuracy, format, description_accuracy, description_format) independently,
   then combine them with configurable weights. This prevents high-variance
   components from drowning out low-variance signal.
"""

import re
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.core_algos import compute_kl
from verl.trainer.metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage
from verl.utils.logger import Tracker
from verl.utils.py_functional import convert_dict_to_str, timer, unflatten_dict

SECOND_HOP_MAX_PROMPT_LENGTH = 2048

SELF_REWARD_VERIFY_PROMPT = (
    "Text description: {Description}\n"
    "Question: {Question}\n"
    "You are provided a text description of a problem and a question. "
    "Determine the answer to the question based on the text description. "
    "First provide an internal step-by-step reasoning within <think> </think> "
    "tags, then provide a single word or phrase answer in \\boxed{{}}."
)

MULTI_ADV_WEIGHTS = {
    "accuracy": 1.0,
    "format": 0.1,
    "description_accuracy": 1.0,
    "description_format": 0.1,
}


def extract_description(text: str) -> str:
    """Extract content between <description> tags, or return full text if absent."""
    match = re.search(r"<description>([\s\S]*?)</description>", text, re.DOTALL)
    if not match:
        return text
    return match.group(1).strip()


def grpo_advantage_for_scores(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute GRPO per-group normalized advantage for a vector of scalar scores.

    Same algorithm as compute_grpo_outcome_advantage but operates on
    pre-summed scalar scores (one per response) instead of token-level rewards.
    Each group (identified by index/uid) is independently normalized to zero
    mean and unit variance.
    """
    id2score: dict[str, list] = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    id2mean, id2std = {}, {}
    for idx, vals in id2score.items():
        t = torch.stack(vals)
        id2mean[idx] = t.mean()
        id2std[idx] = t.std()

    normed = scores.clone()
    for i in range(bsz):
        normed[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    return normed.unsqueeze(-1) * response_mask


class SelfRewardTrainer(RayPPOTrainer):
    """RayPPOTrainer with Vision-SR1 self-reward and multi-advantage GRPO."""

    def __init__(self, *args, reward_component_weights: Optional[dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_component_weights = reward_component_weights or MULTI_ADV_WEIGHTS.copy()

    # ------------------------------------------------------------------
    # Multi-Advantage GRPO
    # ------------------------------------------------------------------

    def _compute_multi_advantage(
        self,
        batch: DataProto,
        reward_metrics_raw: dict[str, list[float]],
    ) -> None:
        """Compute per-component GRPO advantages and combine with weights.

        For each reward component (accuracy, format, description_accuracy,
        description_format), compute a GRPO advantage independently —
        normalizing per-group (uid). The final advantage is a weighted sum.
        This also sets token_level_rewards for metric logging.
        """
        response_mask = batch.batch["response_mask"]
        index = batch.non_tensor_batch["uid"]

        combined_advantage = torch.zeros_like(response_mask, dtype=torch.float32)
        active_components = []
        component_advs = {}

        for component, weight in self.reward_component_weights.items():
            if weight == 0.0 or component not in reward_metrics_raw:
                continue

            scores = torch.tensor(reward_metrics_raw[component], dtype=torch.float32)
            adv = grpo_advantage_for_scores(scores, response_mask, index)
            combined_advantage = combined_advantage + weight * adv
            active_components.append(component)
            component_advs[component] = (scores, adv)

        batch.batch["advantages"] = combined_advantage
        batch.batch["returns"] = combined_advantage

        if active_components:
            print(f"[Multi-Advantage] Components: {active_components}, "
                  f"weights: {[self.reward_component_weights[c] for c in active_components]}")

        n_examples = min(4, len(reward_metrics_raw.get("accuracy", [])))
        for comp in ("accuracy", "description_accuracy"):
            if comp not in component_advs:
                continue
            scores, adv = component_advs[comp]
            per_sample_adv = adv.sum(dim=-1)
            w = self.reward_component_weights[comp]
            print(f"  [{comp}] scores={scores[:n_examples].tolist()}, "
                  f"adv(normed)={per_sample_adv[:n_examples].tolist()}, weight={w}")

    # ------------------------------------------------------------------
    # Second-hop generation
    # ------------------------------------------------------------------

    def _generate_second_hop(
        self,
        gen_batch_output: DataProto,
        questions: list[str],
        n: int,
    ) -> list[str]:
        """Generate text-only second-hop responses from visual descriptions.

        After the first rollout produces responses with <description> tags,
        this method:
        1. Decodes first-rollout responses
        2. Extracts the <description> from each
        3. Builds a text-only verification prompt (description + question)
        4. Generates a second rollout without images (n=1)
        5. Returns decoded second-hop response strings

        Args:
            gen_batch_output: First rollout output (B*n entries).
            questions: Original question texts (B entries).
            n: Number of responses per prompt in the first rollout.

        Returns:
            List of B*n decoded second-hop response strings.
        """
        with torch.no_grad():
            resp_ids = gen_batch_output.batch["responses"]
            if resp_ids.dim() == 3:
                resp_ids = resp_ids.view(-1, resp_ids.size(-1))

            resp_texts = self.tokenizer.batch_decode(
                resp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            B = len(questions)
            total = len(resp_texts)

            if n > 1 and total == B * n:
                questions_expanded = [q for q in questions for _ in range(n)]
            else:
                questions_expanded = list(questions)

            second_hop_prompts = []
            for resp_text, question in zip(resp_texts, questions_expanded):
                description = extract_description(resp_text)
                prompt_text = SELF_REWARD_VERIFY_PROMPT.format(
                    Description=description,
                    Question=question.replace("<image>", "").strip(),
                )
                second_hop_prompts.append(prompt_text)

            max_prompt_len = min(SECOND_HOP_MAX_PROMPT_LENGTH, self.config.data.max_prompt_length)
            chat_prompts = []
            raw_ids_list = []
            for txt in second_hop_prompts:
                messages = [{"role": "user", "content": txt}]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                chat_prompts.append(chat_prompt)
                raw_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)
                if len(raw_ids) > max_prompt_len:
                    raw_ids = raw_ids[:max_prompt_len]
                raw_ids_list.append(raw_ids)

            orig_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer(
                chat_prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = orig_padding_side

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            position_ids = torch.clip(attention_mask.cumsum(dim=1) - 1, min=0)
            raw_prompt_ids = np.array(raw_ids_list, dtype=object)

            meta_info = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            second_gen_batch = DataProto.from_single_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "raw_prompt_ids": raw_prompt_ids,
                },
                meta_info=meta_info,
            )

            second_gen_batch, pad_size = pad_dataproto_to_divisor(
                second_gen_batch, self.actor_rollout_ref_wg.world_size
            )

            second_out = self.actor_rollout_ref_wg.generate_sequences(second_gen_batch)
            second_out = unpad_dataproto(second_out, pad_size=pad_size)

            sec_resp_ids = second_out.batch["responses"]
            if sec_resp_ids.dim() == 3:
                sec_resp_ids = sec_resp_ids.view(-1, sec_resp_ids.size(-1))

            second_texts = self.tokenizer.batch_decode(
                sec_resp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print(f"[Self-Reward] 1st-hop example: {resp_texts[0][:200]}...")
            print(f"[Self-Reward] 2nd-hop input:   {second_hop_prompts[0][:200]}...")
            print(f"[Self-Reward] 2nd-hop output:  {second_texts[0][:200]}...")

            return second_texts

    # ------------------------------------------------------------------
    # Validation with second-hop
    # ------------------------------------------------------------------

    def _validate(self) -> dict[str, Any]:
        """Validation with second-hop generation for description accuracy."""
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation (with self-reward)...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_ref_wg.world_size
            )
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch, pad_size=pad_size * repeat_times
            )

            questions = test_batch.non_tensor_batch.get("question")
            if questions is not None:
                questions_list = questions.tolist()
                desc_answers = self._generate_second_hop(
                    test_output_gen_batch, questions_list, repeat_times
                )
            else:
                desc_answers = [""] * len(test_output_gen_batch)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch.non_tensor_batch["description_answers"] = np.array(
                desc_answers, dtype=object
            )
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(
                self.val_reward_fn.compute_reward.remote(test_batch)
            )

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self._maybe_save_responses(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {
            f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()
        }
        val_length_metrics = {
            f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()
        }
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    # ------------------------------------------------------------------
    # Batch construction with second-hop
    # ------------------------------------------------------------------

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        """Build a training batch with self-reward second-hop generation."""
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        n = self.config.worker.rollout.n
        print("Start generating batch (with self-reward)...")

        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            questions = new_batch.non_tensor_batch["question"].tolist()
            description_answers = self._generate_second_hop(gen_batch_output, questions, n)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=n, interleave=True)

            new_batch.non_tensor_batch["description_answers"] = np.array(
                description_answers, dtype=object
            )

            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low
                    and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. "
                        "Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: rollout_batch_size * n]

    # ------------------------------------------------------------------
    # Training loop with multi-advantage GRPO
    # ------------------------------------------------------------------

    def fit(self):
        """Training loop with multi-advantage GRPO.

        Identical to the base RayPPOTrainer.fit() except the advantage
        computation section: instead of computing a single GRPO advantage
        from the overall reward, we compute per-component advantages
        independently and combine them with configurable weights.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                self._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1
                ).tolist()

                # ── Compute reward (async) ──
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                # ── Multi-advantage computation ──
                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics_raw = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor

                        reduced = {
                            f"reward/{k}": v
                            for k, v in reduce_metrics(reward_metrics_raw).items()
                        }
                        metrics.update(reduced)

                        # KL handling: set token_level_rewards for metrics
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, self.kl_ctrl, self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Per-component GRPO advantages
                        self._compute_multi_advantage(batch, reward_metrics_raw)

                    else:
                        # Fallback: online-filtering pre-computed rewards, single advantage
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, self.kl_ctrl, self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")
