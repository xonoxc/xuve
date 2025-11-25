def safe_text(s: str) -> str:
    return s.encode("latin-1", errors="replace").decode(
        "utf-8",
        errors="replace",
    )
