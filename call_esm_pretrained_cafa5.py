from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from Bio import SeqIO

from constants import TRAIN_FASTA_FPATH, TRAIN_TERMS_FPATH, OBO_FPATH, MODEL_ID, MINIMUM_PROBABILITY


class ProteinFunctionPredictor:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, unique_terms: List[str], go_terms: List[dict]):
        self.model = model
        self.tokenizer = tokenizer
        self.unique_terms = unique_terms
        self.go_terms = go_terms

    @classmethod
    def from_files(cls, model_id: str = MODEL_ID, fasta_fpath: str = TRAIN_FASTA_FPATH, tsv_fpath: str = TRAIN_TERMS_FPATH, obo_fpath: str = OBO_FPATH):
        fasta_data = {}
        for record in SeqIO.parse(fasta_fpath, "fasta"):
            fasta_data[record.id] = str(record.seq)
        tsv_data = cls.parse_tsv_file(tsv_fpath)
        unique_terms = list(set(term for terms in tsv_data.values() for term in terms))

        go_terms = cls.parse_obo_file(obo_fpath)  # Replace with your path
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # First, we load the underlying base model
        base_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        # Then, we load the model with PEFT
        model = PeftModel.from_pretrained(base_model, model_id)

        return cls(model, tokenizer, unique_terms, go_terms)

    @staticmethod
    def parse_tsv_file(file_path: str) -> Dict[str, List[str]]:
        tsv_data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")
                tsv_data[parts[0]] = parts[1:]
        return tsv_data

    @staticmethod
    def parse_obo_file(file_path: str) -> List[dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read().split("[Term]")

        terms = []
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
            terms.append(term)
        return terms

    # 3. The predict_protein_function function
    def classify(self, sequence: str) -> List[str]:
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1022)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
            predicted_indices = torch.where(predictions > MINIMUM_PROBABILITY)[1].tolist()
        
        functions = []
        for idx in predicted_indices:
            term_id = self.unique_terms[idx]  # Use the unique_terms list from your training script
            for term in self.go_terms:
                if term["id"] == term_id:
                    functions.append(term["name"])
                    break

        return functions
    