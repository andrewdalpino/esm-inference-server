from copy import copy

from collections import defaultdict

from typing import Any

import torch

from esm.tokenization import EsmSequenceTokenizer

from esmc_function_classifier.model import EsmcGoTermClassifier

from networkx import DiGraph

import networkx as nx


class GoTermClassifier:
    AVAILABLE_MODELS = {
        "andrewdalpino/ESMC-300M-Protein-Function",
        "andrewdalpino/ESMC-600M-Protein-Function",
    }

    def __init__(
        self, model_name: str, graph: DiGraph, context_length: int, device: str
    ):
        """
        Args:
            model_name: HuggingFace model identifier for the ESMC model.
            context_length: Maximum length of the input sequence.
            device: Device to run the model on (e.g., "cuda" or "cpu").
        """

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "The provided gene ontology must be a directed acyclic graph (DAG)."
            )

        if context_length <= 0:
            raise ValueError("Context length must be greater than 0.")

        tokenizer = EsmSequenceTokenizer()

        model = EsmcGoTermClassifier.from_pretrained(model_name).to(device)

        model.eval()

        self.tokenizer = tokenizer
        self.graph = graph
        self.model = model
        self.context_length = context_length
        self.device = device

    @torch.no_grad()
    def predict_terms(self, sequence: str, top_p: float = 0.5) -> dict[str, Any]:
        """
        Get the GO term probabilities for a given protein sequence.

        Args:
            sequence: The protein sequence to classify.

        Returns:
            Dictionary with the results.
        """

        out = self.tokenizer(
            sequence,
            max_length=self.context_length,
            truncation=True,
        )

        input_ids = out["input_ids"]

        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(self.device)
        )

        y_pred = self.model.forward(input_ids)

        y_prob = torch.sigmoid(y_pred).squeeze(0).tolist()

        probabilities = {
            self.model.id2label[str(index)]: probability
            for index, probability in enumerate(y_prob)
            if probability > top_p
        }

        return {
            "probabilities": probabilities,
        }

    def predict_subgraph(self, sequence: str, top_p: float = 0.5):
        """
        Get the GO subgraph for a given protein sequence.

        Args:
            sequence: The protein sequence to classify.

        Returns:
            Dictionary with the results.
        """

        out = self.predict_terms(sequence, top_p)

        probabilities = defaultdict(float, out["probabilities"])

        # Fix up the predictions by leveraging the GO DAG hierarchy.
        for go_term, parent_probability in copy(probabilities).items():
            for descendant in nx.descendants(self.graph, go_term):
                child_probability = probabilities[descendant]

                probabilities[descendant] = max(
                    parent_probability,
                    child_probability,
                )

        subgraph = self.graph.subgraph(probabilities.keys())

        return {
            "subgraph": subgraph,
            "probabilities": probabilities,
        }
