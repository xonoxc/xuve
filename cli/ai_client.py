from google import genai

from config.data import SYSTEM_PROMPT
from config.env import GEMINI_API_KEY


client = genai.Client(api_key=GEMINI_API_KEY)


def main():
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=SYSTEM_PROMPT
    )

    print(response.text)


if __name__ == "__main__":
    main()
