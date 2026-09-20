import abc
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer


@dataclass
class NMTRawOutput:
    """Raw variables and tensor outputs from an NMT sequence-to-sequence model."""
    loss: float
    perplexity: float
    src_raw_tokens: List[str]
    tgt_raw_tokens: List[str]
    target_ids: List[int]
    token_losses: List[float]
    top_1_token_ids: List[int]
    top_1_tokens: List[str]
    top_1_losses: List[float]
    cross_attentions: np.ndarray  # [num_tgt_subtokens, num_src_subtokens]


@dataclass
class FluencyRawOutput:
    """Raw variables and tensor outputs from a causal language model."""
    loss: float
    perplexity: float
    tokens: List[str]
    token_losses: List[float]
    top_1_tokens: List[str]
    top_1_losses: List[float]


# --- Base Wrappers ---

class SemanticModelWrapper(abc.ABC):
    @abc.abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encodes a list of strings into normalized sentence embeddings."""
        pass


class NMTModelWrapper(abc.ABC):
    @abc.abstractmethod
    def infer(self, source_text: str, target_text: str) -> NMTRawOutput:
        """
        Runs tokenizer and forward pass for source and target text.
        Extracts raw cross-attention matrices, token losses, and top-1 predictions.
        Does not perform word-level breakdown, delta_L thresholding, or alignment mapping.
        """
        pass


class FluencyModelWrapper(abc.ABC):
    @abc.abstractmethod
    def infer(self, text: str) -> FluencyRawOutput:
        """
        Runs tokenizer and causal forward pass on the text.
        Extracts raw shifted token losses, top-1 predictions, and perplexity.
        Does not perform word-level breakdown or error thresholding.
        """
        pass


# --- Concrete Implementations ---

class CamembertSemanticModel(SemanticModelWrapper):
    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        print(f"Loading Semantic Model (sentence-camembert-base) on {self.device}...")
        self.model = SentenceTransformer("dangvantuan/sentence-camembert-base", device=self.device)

    def encode(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=True, device=self.device)


class HelsinkiNMTModel(NMTModelWrapper):
    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        print(f"Loading NMT Model (Helsinki-NLP/opus-mt-es-fr) on {self.device}...")
        self.model_name = "Helsinki-NLP/opus-mt-es-fr"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation="eager"
        ).to(self.device)

    def infer(self, source_text: str, target_text: str) -> NMTRawOutput:
        inputs = self.tokenizer(source_text, return_tensors="pt").to(self.device)
        labels = self.tokenizer(text_target=target_text, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, labels=labels, output_attentions=True)
            loss = float(outputs.loss.item())
            perplexity = float(np.exp(loss))

            if outputs.cross_attentions is not None and len(outputs.cross_attentions) > 0:
                top_layers = outputs.cross_attentions[-4:]
                subtoken_attn = torch.mean(torch.stack(top_layers), dim=(0, 1, 2)).squeeze(0).cpu().float().numpy()
            else:
                subtoken_attn = np.zeros((labels.shape[1], inputs["input_ids"].shape[1]))

            logits = outputs.logits[0].float()  # Float32 for stable loss calculation
            target_ids_tensor = labels[0]
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses_tensor = loss_fct(logits, target_ids_tensor)
            top_1_token_ids_tensor = torch.argmax(logits, dim=-1)
            top_1_losses_tensor = loss_fct(logits, top_1_token_ids_tensor)

        src_raw_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tgt_raw_tokens = self.tokenizer.convert_ids_to_tokens(labels[0])

        target_ids = [int(t.item()) for t in target_ids_tensor]
        token_losses = [float(l.item()) for l in token_losses_tensor]
        top_1_token_ids = [int(t.item()) for t in top_1_token_ids_tensor]
        top_1_tokens = [self.tokenizer.convert_ids_to_tokens(t_id) for t_id in top_1_token_ids]
        top_1_losses = [float(l.item()) for l in top_1_losses_tensor]

        return NMTRawOutput(
            loss=loss,
            perplexity=perplexity,
            src_raw_tokens=src_raw_tokens,
            tgt_raw_tokens=tgt_raw_tokens,
            target_ids=target_ids,
            token_losses=token_losses,
            top_1_token_ids=top_1_token_ids,
            top_1_tokens=top_1_tokens,
            top_1_losses=top_1_losses,
            cross_attentions=subtoken_attn
        )


class GptFrFluencyModel(FluencyModelWrapper):
    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        print(f"Loading Fluency Model (gpt-fr-cased-small) on {self.device}...")
        self.model_name = "asi/gpt-fr-cased-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype
        ).to(self.device)

    def infer(self, text: str) -> FluencyRawOutput:
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids.to(self.device)
        if input_ids.shape[1] == 0:
            return FluencyRawOutput(
                loss=0.0,
                perplexity=100.0,
                tokens=[],
                token_losses=[],
                top_1_tokens=[],
                top_1_losses=[]
            )

        with torch.inference_mode():
            outputs = self.model(input_ids, labels=input_ids)
            loss = float(outputs.loss.item())
            perplexity = float(np.exp(loss))

            logits = outputs.logits[0].float()
            if input_ids.shape[1] > 1:
                shift_logits = logits[:-1, :]
                shift_labels = input_ids[0, 1:]
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses_tensor = loss_fct(shift_logits, shift_labels)
                top_1_ids_tensor = torch.argmax(shift_logits, dim=-1)
                top_1_losses_tensor = loss_fct(shift_logits, top_1_ids_tensor)

                tokens = []
                token_losses = []
                top_1_tokens = []
                top_1_losses = []

                for idx in range(len(shift_labels)):
                    lbl_id = int(shift_labels[idx].item())
                    tok_str = self.tokenizer.decode([lbl_id]).strip()
                    top_1_id = int(top_1_ids_tensor[idx].item())
                    top_1_str = self.tokenizer.decode([top_1_id]).strip()

                    tokens.append(tok_str if tok_str else f"token_{idx}")
                    token_losses.append(float(token_losses_tensor[idx].item()))
                    top_1_tokens.append(top_1_str)
                    top_1_losses.append(float(top_1_losses_tensor[idx].item()))
            else:
                tokens = []
                token_losses = []
                top_1_tokens = []
                top_1_losses = []

        return FluencyRawOutput(
            loss=loss,
            perplexity=perplexity,
            tokens=tokens,
            token_losses=token_losses,
            top_1_tokens=top_1_tokens,
            top_1_losses=top_1_losses
        )


# --- Concrete Implementations ---

class CEFRClassifierModelWrapper(abc.ABC):
    @abc.abstractmethod
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Runs CEFR sequence classification on a list of French texts.
        Returns a list of dicts with:
            level: 'A1' | 'A2' | 'B1' | 'B2' | 'C1' | 'C2'
            score: float (1.0 to 6.0 continuous expectation)
            probabilities: dict[str, float]
        """
        pass


class UniversalCEFRModel(CEFRClassifierModelWrapper):
    def __init__(self, device: str, dtype: torch.dtype, model_name: str = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        print(f"Loading CEFR Classifier Model ({self.model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype
        ).to(self.device)
        self.weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        results = []
        for p in probs:
            pred_idx = int(torch.argmax(p).item())
            pred_level = self.model.config.id2label.get(pred_idx, "A1")
            exp_score = float(torch.dot(p, self.weights).item())
            prob_dict = {
                self.model.config.id2label.get(i, f"L{i}"): float(p[i].item())
                for i in range(len(p))
            }
            results.append({
                "level": pred_level,
                "score": exp_score,
                "probabilities": prob_dict
            })
        return results


# --- Registry ---

class ModelRegistry:
    def __init__(self):
        self._semantic_models = {}
        self._nmt_models = {}
        self._fluency_models = {}
        self._cefr_models = {}

    def register_semantic(self, name: str, cls):
        self._semantic_models[name] = cls

    def register_nmt(self, name: str, cls):
        self._nmt_models[name] = cls

    def register_fluency(self, name: str, cls):
        self._fluency_models[name] = cls

    def register_cefr(self, name: str, cls):
        self._cefr_models[name] = cls

    def get_semantic(self, name: str, device: str, dtype: torch.dtype) -> SemanticModelWrapper:
        if name not in self._semantic_models:
            raise ValueError(f"Semantic model '{name}' not found in registry.")
        return self._semantic_models[name](device, dtype)

    def get_nmt(self, name: str, device: str, dtype: torch.dtype) -> NMTModelWrapper:
        if name not in self._nmt_models:
            raise ValueError(f"NMT model '{name}' not found in registry.")
        return self._nmt_models[name](device, dtype)

    def get_fluency(self, name: str, device: str, dtype: torch.dtype) -> FluencyModelWrapper:
        if name not in self._fluency_models:
            raise ValueError(f"Fluency model '{name}' not found in registry.")
        return self._fluency_models[name](device, dtype)

    def get_cefr(self, name: str, device: str, dtype: torch.dtype) -> CEFRClassifierModelWrapper:
        if name not in self._cefr_models:
            raise ValueError(f"CEFR model '{name}' not found in registry.")
        return self._cefr_models[name](device, dtype)


registry = ModelRegistry()
registry.register_semantic("camembert", CamembertSemanticModel)
registry.register_nmt("helsinki", HelsinkiNMTModel)
registry.register_fluency("gpt_fr", GptFrFluencyModel)
registry.register_cefr("universal_cefr", UniversalCEFRModel)
