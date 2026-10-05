"""How a text is turned into the token ids the analyses feed the model.

The tokenizer is feature-extractor's, which is tokenizer-tools' (shared with
corpus-tools, which counts the bigrams): its special tokens are used as they
are. GPT-2 gets <|endoftext|> first, as in the published code; rinna's
Japanese GPT-2 gets the text alone (lowercased), as in its model card.
"""

MAX_LENGTH = 1024


def encode(tokenizer, text: str, max_length: int = MAX_LENGTH) -> list[int]:
    """Token ids of `text` with the tokenizer's special tokens, truncated to
    `max_length`."""
    return tokenizer(text)["input_ids"][:max_length]


def token_text(tokenizer, token_id: int) -> str:
    """Token as written in the paper, with "_" for a leading space (GPT-2's
    "Ġ" and SentencePiece's "▁")."""
    return tokenizer.convert_ids_to_tokens(token_id).replace("Ġ", "_").replace("▁", "_")
