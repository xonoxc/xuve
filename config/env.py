import os
from dotenv import load_dotenv


load_dotenv()
# loading the ai api key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
