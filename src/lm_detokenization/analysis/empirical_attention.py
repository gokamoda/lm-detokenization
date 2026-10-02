"""First-layer attention weights observed on natural text (Section 5.4)."""

from dataclasses import dataclass
from pathlib import Path

import torch
from feature_extractor import FeatureExtractor
from feature_extractor.configs import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_LENGTH = 1024


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
    """Layer-0 attention rows at `positions`. The tokenizer of feature-extractor
    prepends <|endoftext|> for GPT-2, so position 0 is that token; texts are
    truncated to MAX_LENGTH tokens."""
    extractor = FeatureExtractor(model_name)
    extractor.configure(FeatureConfig.from_str(["attn.layer_00.attn_weights"]))
    data_loader = DataLoader(
        TextDataset([TextDataEntry(idx=str(k), text=t) for k, t in enumerate(texts)]),
        batch_size=1,  # no padding inside the extracted attention
        collate_fn=create_collator(extractor.tokenizer, max_length=MAX_LENGTH),
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
