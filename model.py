import torch

from torch import Tensor

from abc import ABC

from transformers import AutoTokenizer, EsmForSequenceClassification

from typing import Any

from util import Timer


class ESMModel:
    def __init__(self, name: str, logit_processor: "LogitProcessor", device: str):
        """
        Initialize the ESM model.

        Args:
            name: HuggingFace model identifier for the ESM model.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = EsmForSequenceClassification.from_pretrained(name).to(device)
        self.logit_processor = logit_processor

        self.device: str = device

    @torch.no_grad()
    def classify(self, sequence: str) -> dict[str, Any]:
        """
        Get the classifications for a protein sequence.

        Args:
            sequence: The protein sequence to classify.

        Returns:
            Dictionary with classification results.
        """

        inputs = self.tokenizer(sequence, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with Timer() as timer:
            outputs = self.model(**inputs)

            result = self.logit_processor.process(outputs.logits.squeeze())

        result["runtime"] = timer.duration

        return result


class LogitProcessor(ABC):
    def process(self, logits: Tensor) -> dict[str, Any]:
        """
        Process the logits output from the model.

        Args:
            logits: The raw logits output from the model.

        Returns:
            Dictionary with processed results.
        """

        raise NotImplementedError("Subclasses should implement this method.")


class ProbabilitiesLogitProcessor(LogitProcessor):
    """Logit processor that converts logits to probabilities."""

    @torch.no_grad()
    def process(self, logits: Tensor) -> dict[str, Any]:
        probabilities = torch.sigmoid(logits).squeeze()

        return {
            "probabilities": probabilities.tolist(),
        }


class ProteinFunctionLogitProcessor(LogitProcessor):
    """Logit processor for protein function classification using GO terms."""

    @classmethod
    def from_files(
        cls, tsv_fpath: str, obo_fpath: str, min_probability: float
    ) -> "ProteinFunctionLogitProcessor":
        tsv_data = cls.parse_tsv_file(tsv_fpath)

        unique_terms = list(set(term for terms in tsv_data.values() for term in terms))

        go_terms = cls.parse_obo_file(obo_fpath)

        return cls(unique_terms, go_terms, min_probability)

    @staticmethod
    def parse_tsv_file(file_path: str) -> dict[str, dict[str]]:
        tsv_data = {}

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                tsv_data[parts[0]] = parts[1:]

        return tsv_data

    @staticmethod
    def parse_obo_file(file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read().split("[Term]")

        go_terms_by_id = {}

        for entry in data[1:]:
            lines = entry.strip().split("\n")

            term = {}

            for line in lines:
                if line.startswith("id:"):
                    term["id"] = line.split("id:")[1].strip()
                elif line.startswith("name:"):
                    term["name"] = line.split("name:")[1].strip()
                elif line.startswith("namespace:"):
                    term["namespace"] = line.split("namespace:")[1].strip()
                elif line.startswith("def:"):
                    term["definition"] = line.split("def:")[1].split('"')[1]

            go_terms_by_id[term["id"]] = term

        return go_terms_by_id

    def __init__(
        self, unique_terms: list[str], go_terms: dict[str, dict], min_probability: float
    ):
        if min_probability < 0 or min_probability > 1:
            raise ValueError("min_probability must be between 0 and 1.")

        self.unique_terms = unique_terms
        self.go_terms = go_terms
        self.min_probability = min_probability

    @torch.no_grad()
    def process(self, logits: Tensor) -> dict[str, Any]:
        """Process the logits output from the model."""

        probabilities = torch.sigmoid(logits)

        predicted_indices = torch.where(probabilities > self.min_probability)
        predicted_indices = predicted_indices[0].tolist()

        probabilities = probabilities[predicted_indices].tolist()

        functions = []

        for idx in predicted_indices:
            term_id = self.unique_terms[idx]

            functions.append(self.go_terms[term_id]["name"])

        return {
            "functions": functions,
            "probabilities": probabilities,
        }
