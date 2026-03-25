import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
