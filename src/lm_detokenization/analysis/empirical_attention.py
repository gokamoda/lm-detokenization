"""First-layer attention weights observed on natural text (Section 5.4)."""

from dataclasses import dataclass
from pathlib import Path

import torch
from feature_extractor import FeatureExtractor
from feature_extractor.configs import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from lm_detokenization.tokens import MAX_LENGTH, encode


@dataclass
class AttentionRows:
    """Attention from one query position i to j = 0 .. i, for every document
    long enough to have position i."""

    position: int
    document_index: list[int]
    weights: torch.Tensor  # [document, head, position + 1], float16

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "position": self.position,
                "document_index": self.document_index,
                "weights": self.weights,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "AttentionRows":
        d = torch.load(path, weights_only=True)
        return cls(
            position=d["position"],
            document_index=d["document_index"],
            weights=d["weights"],
        )


def extract_attention_rows(
    texts: list[str], positions: list[int], model_name: str = "gpt2"
) -> dict[int, AttentionRows]:
    """Layer-0 attention rows at `positions`. Texts are tokenized by
    tokens.encode (for GPT-2, position 0 is <|endoftext|>) and truncated to
    MAX_LENGTH tokens."""
    extractor = FeatureExtractor(model_name)
    extractor.configure(FeatureConfig.from_str(["attn.layer_00.attn_weights"]))

    def collate(batch: list[TextDataEntry]) -> dict:
        ids = torch.tensor([encode(extractor.tokenizer, e.text) for e in batch])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "indices": [e.idx for e in batch],
        }

    data_loader = DataLoader(
        TextDataset([TextDataEntry(idx=str(k), text=t) for k, t in enumerate(texts)]),
        batch_size=1,  # no padding inside the extracted attention
        collate_fn=collate,
    )
    rows: dict[int, list] = {p: [] for p in positions}
    index: dict[int, list] = {p: [] for p in positions}
    for batch, features in tqdm(
        extractor.extract_features(data_loader), total=len(texts), desc="attention"
    ):
        attn = features.attn[0].attn_weights[0].cpu()  # [head, L, L]
        for p in positions:
            if attn.shape[-1] > p:
                rows[p].append(attn[:, p, : p + 1].to(torch.float16))
                index[p].append(int(batch["indices"][0]))
    return {
        p: AttentionRows(
            position=p,
            document_index=index[p],
            weights=torch.stack(rows[p]) if rows[p] else torch.empty(0),
        )
        for p in positions
    }
