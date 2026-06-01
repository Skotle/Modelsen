from .byte_tokenizer import ByteTokenizer
from .char_tokenizer import CharTokenizer

Tokenizer = ByteTokenizer | CharTokenizer


def tokenizer_from_dict(values: dict[str, object] | None):
    values = values or {}
    tokenizer_type = values.get("type", ByteTokenizer.name)
    if tokenizer_type == ByteTokenizer.name:
        return ByteTokenizer.from_dict(values)
    if tokenizer_type == CharTokenizer.name:
        return CharTokenizer.from_dict(values)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


__all__ = ["ByteTokenizer", "CharTokenizer", "Tokenizer", "tokenizer_from_dict"]
