import torch

from transformers import AutoTokenizer, EsmForSequenceClassification

from typing import Dict, Any


class ESMModel:
    def __init__(self, name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cpu"):
        """
        Initialize the ESM model.

        Args:
            name: HuggingFace model identifier for the ESM model.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = EsmForSequenceClassification.from_pretrained(name).to(device)

        self.device: str = device

    def classify(self, sequence: str) -> Dict[str, Any]:
        """
        Get the classifications for a protein sequence.

        Args:
            sequence: The protein sequence to classify.

        Returns:
            Dictionary with classification results.
        """

        inputs = self.tokenizer(sequence, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            probabilities = torch.sigmoid(outputs.logits.squeeze())

        return {
            "probabilities": probabilities.tolist(),
        }
