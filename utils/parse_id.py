from typing import List


def parse_id_list(text: str) -> List[int]:
    import json
    import re

    try:
        return json.loads(text)
    except Exception:
        nums = re.findall(r"\d+", text)
        return [int(n) for n in nums]
