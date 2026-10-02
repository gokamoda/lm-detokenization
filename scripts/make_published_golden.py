"""Dump reference values computed by the code at tag `published`.

Run against a worktree of `published`, with the environment of the time
(transformers 4.x):

    git worktree add --detach <dir> published
    PYTHONPATH=<dir>/src uv run python scripts/make_published_golden.py

The output (outputs/golden/published.pt) is compared with the migrated code
by src/published_golden_test.py.
"""

from pathlib import Path

import torch
from eqmodels import EQGPT2LMHeadModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from detokenization.socres_and_counts import DetokenizationDataRetriever
from empirical.six_terms_importance import compute_6terms

MODEL_NAME = "gpt2"
SAVE_PATH = Path("outputs/golden/published.pt")

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Homo sapiens is the only extant species of the genus Homo.",
    "In 1905, Albert Einstein published four groundbreaking papers that "
    "changed the course of modern physics, including the special theory of "
    "relativity and the photoelectric effect.",
]
DETOK_QUERIES = [" sapiens", " Einstein", "ization"]


def main():
    model = EQGPT2LMHeadModel.from_pretrained(MODEL_NAME)
    wte = model.transformer.wte.weight.detach().cpu()
    wpe = model.transformer.wpe.weight.detach().cpu()
    wqkh = model.transformer.h[0].attn.wqkh.detach().cpu()
    bqwkh = model.transformer.h[0].attn.bqwkh.detach().cpu()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # empirical/six_terms_importance.py prepends <|endoftext|> by hand
    six_terms = []
    for text in PROMPTS:
        prompt = "<|endoftext|>" + text
        six_terms.append(
            {
                "input_ids": tokenizer(prompt, return_tensors="pt").input_ids,
                "scores": compute_6terms(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    wte=wte,
                    wpe=wpe,
                    w_compare=wqkh,
                    w_self=bqwkh,
                ),
            }
        )

    # detokenization/socres_and_counts.py
    retriever = DetokenizationDataRetriever(model_name=MODEL_NAME)
    detok = {}
    for query in DETOK_QUERIES:
        suffix_id = tokenizer.encode(query)[-1]
        scores = retriever.get_detokenization_scores(
            suffix_id, retriever.wte.unsqueeze(0), retriever.wqkh
        )
        detok[query] = {"suffix_id": suffix_id, "scores": scores.cpu()}

    # empirical/vs_tptpp.py main: layer 0 attention of the original model,
    # prompts tokenized without <|endoftext|>
    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    hf_model.eval()
    vs_tptpp = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = hf_model(**inputs, output_attentions=True)
        vs_tptpp.append(
            {"input_ids": inputs.input_ids, "attn_l0": output.attentions[0][0].cpu()}
        )

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompts": PROMPTS,
            "wqkh": wqkh,
            "bqwkh": bqwkh,
            "six_terms": six_terms,
            "detok": detok,
            "vs_tptpp": vs_tptpp,
        },
        SAVE_PATH,
    )
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
