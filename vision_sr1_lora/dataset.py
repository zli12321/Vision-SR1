from verl.utils.dataset import RLHFDataset


class VisionSR1Dataset(RLHFDataset):
    """RLHFDataset that preserves the raw question text for second-hop generation.

    The parent class pops the prompt_key from the example dict during __getitem__,
    so we save it as 'question' before the parent processes it. This allows the
    trainer to access the original question text when building text-only
    second-hop prompts for self-reward.
    """

    def __getitem__(self, index):
        raw_question = self.dataset[index][self.prompt_key]
        example = super().__getitem__(index)
        example["question"] = raw_question
        return example
