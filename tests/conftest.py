import os
import pytest

# Set a dummy API key for tests before any modules are imported
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_PASSWORD"] = "test-password"
