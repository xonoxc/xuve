from google.genai.errors import ServerError, APIError

from lib.ai.client import LLM_CLIENT
from lib.ai.prompt_builders import build_prompt

from lib.enums.enahnce_methods import EnhanceMethod


def enhance_query(query: str, method: EnhanceMethod | None) -> str:
    if not method:
        return query.strip()

    try:
        response = LLM_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_prompt(
                query.strip(),
                EnhanceMethod(method),
            ),
        )
    except (ServerError, APIError) as e:
        if e.code == 429:
            print(e.message)
        else:
            print(f"Server Error: {e.message}")

        return query.strip()

    enahnced_query = response.text
    if not enahnced_query:
        return query.strip()

    print(f"Enhanced query ({method}): '{query}' -> '{enahnced_query}'\n")

    return enahnced_query.strip()
