import torch

from transformers import AutoTokenizer, EsmTokenizer, EsmForSequenceClassification

from typing import Any

from util import Timer


class ESMClassifier:
    def __init__(self, model_name: str, context_length: int, device: str):
        """
        Initialize the ESM model.

        Args:
            model_name: HuggingFace model identifier for the ESM model.
            logit_processor: Logit processor to convert logits to probabilities.
            context_length: Maximum length of the input sequence.
            device: Device to run the model on (e.g., "cuda" or "cpu").
        """

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = EsmForSequenceClassification.from_pretrained(model_name).to(device)

        model.eval()

        self.tokenizer: EsmTokenizer = tokenizer
        self.model: EsmForSequenceClassification = model
        self.context_length: int = context_length
        self.device: str = device

    @torch.no_grad()
    def predict_rank(self, sequence: str, top_k: int) -> dict[str, Any]:
        """
        Get the top k binary classifications for a protein sequence.

        Args:
            sequence: The protein sequence to classify.
            top_k: The number of top classifications to return.

        Returns:
            Dictionary with the results.
        """

        out = self.tokenizer(
            sequence,
            max_length=self.context_length,
            truncation=True,
        )

        input_ids = out["input_ids"]
        attn_mask = out["attention_mask"]

        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(self.device)
        )
        attn_mask = (
            torch.tensor(attn_mask, dtype=torch.int64).unsqueeze(0).to(self.device)
        )

        with Timer() as timer:
            outputs = self.model(input_ids, attention_mask=attn_mask)

            logits = outputs.logits.squeeze()

            probabilities = torch.sigmoid(logits)

            probabilities, indices = torch.topk(probabilities, top_k)

            probabilities = probabilities.tolist()

            labels = [self.model.config.id2label[index] for index in indices.tolist()]

        return {
            "labels": labels,
            "probabilities": probabilities,
            "meta": {
                "runtime": timer.duration,
            },
        }
