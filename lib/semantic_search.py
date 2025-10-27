from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
        )


def validate_model() -> None:
    MODEL = SemanticSearch().model

    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MODEL.max_seq_length}")
