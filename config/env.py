import os
from dotenv import load_dotenv


load_dotenv()
# loading the ai api key
GEMINI_API_KEY = os.environ.get("GEM_API_KEY")
