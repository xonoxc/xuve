from google.genai.errors import ServerError, APIError

from lib.ai.client import LLM_CLIENT
from lib.enums.enahnce_methods import EnhanceMethod


def build_prompt(query: str) -> str:
    return f"""Fix any spelling errors in this movie search query.

           Only correct obvious typos. Don't change correctly spelled words.

           Query: "{query}"

           If no errors, return the original query.DONT MAKE ANY MISTAKES!!!
           Corrected:"""


def generate_correct_spelling(query: str, method: EnhanceMethod | None) -> str:
    if not method:
        return ""

    try:
        response = LLM_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_prompt(
                query.strip(),
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
