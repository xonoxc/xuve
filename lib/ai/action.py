import time
from google.genai.errors import ServerError, APIError
from google.genai.types import GenerateContentResponse

from lib.ai.client import LLM_CLIENT
from lib.ai.prompt_builders import build_prompt

from lib.enums.enahnce_methods import EnhanceMethod


def generate_resp(
    prompt: str,
    delay: float | None = None,
) -> GenerateContentResponse | None:
    try:
        return LLM_CLIENT.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
    except (ServerError, APIError) as e:
        if e.code == 429:
            print(e.message)
            # just to avoid hitting rate limit
            if delay:
                print(f"retrying after {delay} seconds...")
                time.sleep(delay)
                return generate_resp(
                    prompt,
                    delay,
                )
        else:
            print(f"Server Error: {e.message}")


def enhance_query(
    query: str,
    method: EnhanceMethod | None,
) -> str:
    stripped_query = query.strip()

    if not method:
        return stripped_query

    response = generate_resp(
        build_prompt(
            query=stripped_query,
            method=EnhanceMethod(method),
        )
    )
    if not response:
        return stripped_query

    enahnced_query = response.text
    if not enahnced_query:
        return stripped_query

    print(f"Enhanced query ({method}): '{query}' -> '{enahnced_query}'\n")

    return enahnced_query.strip()
