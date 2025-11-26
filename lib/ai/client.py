from google import genai
from config.env import GEMINI_API_KEY


LLM_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
