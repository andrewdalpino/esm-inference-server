from copy import copy

from collections import defaultdict

from typing import Any

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from networkx import DiGraph

import networkx as nx


class GoTermClassifier:
    AVAILABLE_MODELS = {
        "andrewdalpino/ESM2-35M-Protein-Molecular-Function",
        "andrewdalpino/ESM2-35M-Protein-Biological-Process",
        "andrewdalpino/ESM2-35M-Protein-Cellular-Component",
        "andrewdalpino/ESM2-150M-Protein-Molecular-Function",
        "andrewdalpino/ESM2-150M-Protein-Biological-Process",
        "andrewdalpino/ESM2-150M-Protein-Cellular-Component",
    }

    def __init__(
        self,
        model_name: str,
        graph: DiGraph,
        context_length: int,
        device: str,
        dtype: torch.dtype,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for the ESMC model.
            graph: Directed acyclic graph (DAG) representing the Gene Ontology (GO).
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype
        )

        model = model.to(device)

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

        out = self.model.forward(input_ids)

        y_prob = torch.sigmoid(out.logits).squeeze(0).tolist()

        probabilities = {
            self.model.config.id2label[index]: probability
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
