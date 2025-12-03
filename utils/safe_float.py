def convert_to_float(value: str | None, fallback=0.0) -> float:
    if value is None:
        return fallback

    try:
        return float(value.strip())
    except ():
        return fallback
