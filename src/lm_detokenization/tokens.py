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


def suffix_id(tokenizer, word: str) -> int:
    """The last token of `word`, as the current token (suffix) of a
    detokenization: a token inside a word.

    GPT-2's "iens" is so already ("Ġ" marks a space only if `word` has one,
    as " Jackson"). A SentencePiece tokenizer puts "▁" (a space) before a
    text, so a word of one token is the token at the start of a word
    (rinna/japanese-gpt-1b tokenizes パン as ▁パン); the same token without
    "▁" is taken instead if the vocabulary has it.
    """
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    last = tokenizer.convert_ids_to_tokens(ids[-1])
    if len(ids) == 1 and last.startswith("▁") and len(last) > 1:
        inside = tokenizer.convert_tokens_to_ids(last[1:])
        if inside != tokenizer.unk_token_id:
            return inside
    return ids[-1]
