"""
llamacpp test script
"""
from llama_cpp import Llama

QUERY_GEN_PROMPT = """
You generate Google search queries for football match analysis.

Task:
Generate exactly 3 short search queries for the following request:
"{request}"

Rules:
- Queries must be concise (max 8 words each)
- Do NOT explain anything
- Do NOT add bullet points or numbering
- Output ONLY a JSON array of strings
- DO NOT GENERATE MORE THAN 3 SEARCH QUERIES

Example output:
["query 1",
 "query 2",
 "query 3"]
"""


llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
# Call the model (use small max_tokens while testing to speed things up)
PROMPT=QUERY_GEN_PROMPT.format(request="Arsenal last 5 matches")
resp = llm(PROMPT, max_tokens=128)
queries = resp["choices"][0]["text"]

print(queries)
