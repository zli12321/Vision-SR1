# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


# SequentialRewardFunction = Callable[[str, str], RewardScore]

# BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]

# → prediction, ground-truth, question
SequentialRewardFunction = Callable[[str, str, str], RewardScore]
# → lists of the same three pieces
BatchRewardFunction = Callable[
    [List[str], List[str], List[str]], List[RewardScore]
]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            question     = data.non_tensor_batch["question"][i]
            
            # score = self.reward_fn(response_str, ground_truth)
            score = self.reward_fn(response_str, ground_truth, question)
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


# class BatchFunctionRewardManager(FunctionRewardManager):
#     reward_fn: BatchRewardFunction

#     def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
#         # response_str, ground_truth = [], []
#         response_str, ground_truth, questions = [], [], []
#         response_ids = data.batch["responses"]
#         response_length = data.batch["response_mask"].sum(dim=-1)
        
        
#         for i in range(len(data)):
#             valid_response_ids = response_ids[i][: response_length[i]]
#             response_str.append(
#                 self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
#             )
#             ground_truth.append(data.non_tensor_batch["ground_truth"][i])
#             questions.append(data.non_tensor_batch["question"][i])

#         # scores = self.reward_fn(response_str, ground_truth)
#         scores = self.reward_fn(response_str, ground_truth, questions)
#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_metrics = defaultdict(list)
#         for i, score in enumerate(scores):
#             reward_tensor[i, response_length[i] - 1] = score["overall"]
#             for key, value in score.items():
#                 reward_metrics[key].append(value)

#         return reward_tensor, reward_metrics

# from collections import defaultdict
# from typing import Tuple, Dict, List
# import torch

# class BatchFunctionRewardManager(FunctionRewardManager):
#     reward_fn: BatchRewardFunction

#     def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
#         response_str, ground_truth, questions, description_answers = [], [], [], []

#         response_ids   = data.batch["responses"]
#         response_len   = data.batch["response_mask"].sum(dim=-1)
#         # ➊ pull once to avoid repeated dict look-ups
#         gt_arr         = data.non_tensor_batch["ground_truth"]
#         qn_arr         = data.non_tensor_batch["question"]
#         desc_arr       = data.non_tensor_batch.get("description_answers", None)

#         for i in range(len(data)):
#             valid_ids = response_ids[i][: response_len[i]]
#             response_str.append(
#                 self.tokenizer.decode(
#                     valid_ids, skip_special_tokens=self.config.skip_special_tokens
#                 )
#             )
#             ground_truth.append(gt_arr[i])
#             questions.append(qn_arr[i])

#             # if description_answers provided, align it; else use ""
#             description_answers.append("" if desc_arr is None else desc_arr[i])

#         # ➋ call the 4-arg reward function
#         scores = self.reward_fn(
#             response_str, ground_truth, questions, description_answers
#         )

#         # same tensor ​+ metric aggregation as before
#         reward_tensor  = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_metrics = defaultdict(list)
#         for i, score in enumerate(scores):
#             reward_tensor[i, response_len[i] - 1] = score["overall"]
#             for k, v in score.items():
#                 reward_metrics[k].append(v)

#         return reward_tensor, reward_metrics



from collections import defaultdict
from typing import Tuple, Dict, List
import torch

class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_str, ground_truth, questions, description_answers = [], [], [], []

        response_ids   = data.batch["responses"]
        response_len   = data.batch["response_mask"].sum(dim=-1)
        # ➊ pull once to avoid repeated dict look-ups
        gt_arr         = data.non_tensor_batch["ground_truth"]
        qn_arr         = data.non_tensor_batch["question"]
        desc_arr       = data.non_tensor_batch.get("description_answers", None)

        for i in range(len(data)):
            valid_ids = response_ids[i][: response_len[i]]
            response_str.append(
                self.tokenizer.decode(
                    valid_ids, skip_special_tokens=self.config.skip_special_tokens
                )
            )
            ground_truth.append(gt_arr[i])
            questions.append(qn_arr[i])

            # if description_answers provided, align it; else use ""
            description_answers.append("" if desc_arr is None else desc_arr[i])

        # ➋ call the 4-arg reward function
        scores = self.reward_fn(
            response_str, ground_truth, questions, description_answers
        )

        B, T = data.batch["responses"].shape
        device = data.batch["responses"].device
        token_mask = data.batch["response_mask"]       # (B, T)

        # allocate (B, T, 2)   ← TWO reward channels
        token_level = torch.zeros(B, T, 2,
                                dtype=torch.float32, device=device)


        for i, score in enumerate(scores):
            # last = token_mask[i].sum() - 1             # index of last answer token
            # ── index of the final token in the i-th response
            last = int(response_len[i].item()                # tensor → python int
                    if torch.is_tensor(response_len[i])
                    else response_len[i]) - 1

            token_level[i, last, 0] = score["accuracy"]
            token_level[i, last, 1] = score["description_accuracy"]

        # collect metrics
        metrics = defaultdict(list)
        for s in scores:
            for k, v in s.items():
                metrics[k].append(v)

        return token_level, metrics                    


