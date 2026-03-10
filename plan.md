# Football Match Result Predictor — LLM-ready Implementation Plan (TAVILY as search provider, NO CODE)

This is the **exact same plan** as previously agreed, updated only to replace **Serper.dev** with **Tavily** as the default search + extract provider. The plan remains strictly **NO CODE** — it describes files, responsibilities, prompts, env vars, and step-by-step tasks that the coding LLM must implement. Paste this whole Markdown into your agent.

---

# 1. Project objective (MVP)

Build a Streamlit-based prototype that, for any two user-entered football teams, does the following end-to-end:

1. Generate 3–6 short search queries with a local LLM to find: last 5 results for Team A, last 5 results for Team B, last 5 head-to-head results (if any), and league rankings.  
2. Execute those queries using **Tavily** and collect top results (title, snippets/extracted content, url, any structured fields the provider returns, MAKE SURE TO USE include_answer="advanced" parameter and retrieve the 'answer' field as it has the full clean answer to the search query).  
3. Parse the returned snippets/extracted content (and call Tavily’s Extract endpoint for page-level content only when needed) to construct concise structured features: `teamA.last_5` (W/D/L and optionally scorelines), `teamB.last_5`, `h2h.summary`, and `league_rank` for each team (if present).  
4. Feed this structured summary to the LLM and ask for a single JSON response containing: `winner`, `probability` (0–100), one-sentence `reasoning`, and three `top_signals`.  
5. Display everything in Streamlit for transparency: generated queries, raw Tavily responses, parsed features, and final JSON prediction. Persist a full log per run in `runs/<timestamp>.json`.

**Acceptance criteria (MVP)**:
- End-to-end flow works locally with your quantized GGUF LLM and Tavily API key.  
- LLM outputs valid JSON for both query-generation and prediction prompts.  
- Each run produces a log file in `runs/` containing queries, raw Tavily JSON, parsed features, and the prediction JSON.

---

# 2. High-level architecture & dataflow (textual)

**Components**
- Streamlit UI: single-file app to collect team names and trigger prediction.  
- Orchestrator (runner): coordinates LLM query generation → Tavily search client → parsing → LLM prediction; returns the final prediction and persists a run log.  
- Search adapter: a pluggable interface with a concrete `tavily` implementation that returns canonicalized search/extract result objects: `title`, `snippet` / `extracted_text`, `url`, `source: "tavily"`, `raw`.  
- Parser: heuristics to extract W/D/L or numeric scorelines from Tavily snippets/extracted content; fallback to Tavily Extract for page content when snippets lack signals.  
- LLM client wrapper: stable wrapper for `llama-cpp-python` exposing `generate(prompt, max_tokens)` and normalizing response text extraction.  
- Tests: unit tests for search adapter and LLM wrapper (mocks); integration smoke test for one demo run.

**Workflow (Option B — orchestrated)**
1. UI sends team names to runner.  
2. Runner asks local LLM to produce query array.  
3. Runner uses the Tavily client to execute queries and receives structured results / extracted content.  
4. Runner parses content to build features. If snippets/content lack needed signals, runner calls Tavily Extract for top URL(s).  
5. Runner builds a compact features JSON and calls local LLM for final prediction (JSON-only).  
6. Runner stores full run log and returns the parsed prediction to UI.

**Notes**
- We use **Option B** (backend-controlled orchestration) to control quota usage, caching, and logging.
- Keep runner synchronous for MVP: simple function callable from Streamlit.

---

# 3. Repo layout the agent MUST create (exact names)

- `footy-predictor/`
  - `models/`                — for GGUF models (not committed)
  - `.env`                   — holds `TAVILY_API_KEY` and optional `MODEL_PATH`
  - `env.yml`                — conda environment spec
  - `README.md`              — run & install instructions (including pip package for Tavily)
  - `app.py`                 — Streamlit single-file UI (entrypoint for MVP)
  - `backend/`
    - `__init__.py`
    - `runner.py`            — orchestrator, single function `run_prediction(team_a, team_b)`
    - `search_adapter/`
      - `__init__.py`
      - `base.py`            — adapter interface (`search(query, k)->list`)
      - `tavily.py`          — Tavily implementation (replace previous `serper.py`)
    - `llm/`
      - `__init__.py`
      - `llm_client.py`      — wrapper to the `llama-cpp-python` callable API
    - `parse/`
      - `__init__.py`
      - `parser.py`          — extract features from Tavily results and optional extract fallback
  - `tests/`
    - `test_search_adapter.py`
    - `test_llm_client.py`
  - `runs/`                  — runtime logs (auto-created by runner)

> Note: `tavily.py` replaces the previous `serper.py` file name and responsibilities.

---

# 4. Environment & prerequisites (what the agent should add to README)

**Conda env (agent must add `env.yml` with Python 3.10 and these pip packages)**:
- streamlit  
- requests  
- beautifulsoup4  
- python-dotenv  
- llama-cpp-python  
- pandas  
- pytest  
- playwright  
- tiktoken  
- **tavily-python**  ← **install this** to use the Tavily client  
- (optional) fastapi, uvicorn for a future API layer

**System-level notes**
- On Windows, recommend WSL2 for easier `llama.cpp` tooling.  
- After env creation, run: `python -m playwright install` (Playwright is needed only for any optional local page-render fallback, but Tavily Extract will usually avoid this).

---

# 5. Detailed file responsibilities (what the agent must implement — plain English)

1. **backend/search_adapter/base.py**  
   - Define `SearchAdapter` interface: `search(query: str, k: int) -> List[dict]`.  
   - Canonical dict keys: `title`, `snippet` or `extracted_text`, `url`, `source`, `raw`.  
   - On HTTP errors: raise descriptive exception; on zero results: return empty list.

2. **backend/search_adapter/tavily.py**  
   - Implement `TavilyAdapter` class conforming to `SearchAdapter`.  
   - Read `TAVILY_API_KEY` from environment (via `.env`).  
   - Call Tavily Search endpoint with params: query text, count/max_results, optional search depth; normalize the response to canonical list and include provider `raw`.  
   - If results include extracted content (Tavily may return sanitized page text or structured fields), prioritize that in `extracted_text`.  
   - Implement simple retry logic for transient network errors (3 attempts, exponential backoff).  
   - On quota or 4xx/5xx errors, raise a helpful error that includes the provider status and message.  
   - Unit tests must mock Tavily SDK / HTTP responses and ensure normalization and retry behavior.

3. **backend/llm/llm_client.py**  
   - Implement `LocalLLM` wrapper accepting `model_path` and `n_ctx`.  
   - Provide `generate(prompt: str, max_tokens: int)` returning the trimmed main text.  
   - Handle response shapes (choices/text/fallback) robustly.  
   - If model-load/memory problems occur, raise a descriptive exception with actionable suggestions (reduce `n_ctx`, use smaller model).  
   - Unit test: mock Llama object and assert return extraction.

4. **backend/parse/parser.py**  
   - `extract_snippet_features(results_by_query: dict) -> features_dict`:  
     - Input: mapping `query` → list of Tavily-normalized dicts.  
     - Output schema: `teamA.last_5` (list of `"W"|"D"|"L"` optionally with `scores`), `teamB.last_5`, `h2h.summary`, `league_ranks`.  
     - Heuristics: find `\d+-\d+` patterns, verbs like `beat`, `drew`, `lost`, dates. If provider returned `extracted_text`, parse it first.  
   - `fetch_extracted_content(urls: list) -> dict` (adapter to call Tavily Extract):  
     - When snippets/content lack signals, call Tavily Extract for top URL(s) and use provider-sanitized content. Keep it conservative (top-1 per deficient query).  
     - Timeouts short; failures lead to empty string and logged warning.

5. **backend/runner.py**  
   - `run_prediction(team_a: str, team_b: str) -> dict` steps:
     1. Validate & normalize inputs.  
     2. Build the query-generation prompt (authoritative template below) and call `LocalLLM.generate`. Parse returned JSON array (robustly).  
     3. For each query, call `TavilyAdapter.search(query, k=5)` and save raw responses.  
     4. Build features via `extract_snippet_features`. If insufficient signals (rule: fewer than 3 explicit outcomes per team OR missing ranks), call Tavily Extract for top URL(s) and re-run parsing.  
     5. Build prediction prompt (authoritative template below), call LLM, robustly parse JSON response, validate fields `winner`, `probability`, `reasoning`, `top_signals`. If validation fails, one corrective retry with a strict JSON-only instruction.  
     6. Persist run log `runs/run_<timestamp>.json` containing teams, timestamp, queries, raw_tavily, features, prompts_used, llm_responses, parsed_prediction.  
     7. Return parsed prediction to caller.  
   - Error handling: wrap provider errors, quota messages, parsing failures and return friendly messages to UI.

6. **app.py (Streamlit)**  
   - Single-page UI with Team A and Team B inputs, Predict button, spinner while running.  
   - On predict: call `runner.run_prediction`, display:
     - Generated queries (expandable).  
     - Raw Tavily results grouped by query (title, snippet/extracted_text, url).  
     - Parsed features JSON (pretty).  
     - Final prediction JSON and human-friendly summary (e.g., "Predicted winner: Arsenal — 68%").  
   - Show link/button to open latest run log. Input sanitization and friendly error display required.

7. **tests/**  
   - `test_search_adapter.py`: mock Tavily SDK / HTTP and assert normalization.  
   - `test_llm_client.py`: mock Llama interface and assert `generate()` extraction.  
   - Optional integration smoke test that uses mocked Tavily responses and runs `run_prediction`.

---

# 6. Prompt templates — exact text (use verbatim)

**Query generation prompt** (replace `{request}`):
You are a concise query generator.

Task:
Given the request: "{request}"

Output EXACTLY a JSON array (no commentary), containing 3 to 6 short Google-style search queries that retrieve:

last 5 match results for team A,

last 5 match results for team B,

head-to-head matches,

league ranking pages.

Rules:

Each query <= 8 words.

Do not add explanations or text outside the JSON array.

Example output:
["Arsenal last 5 matches results", "Arsenal fixtures last 5", "Arsenal vs Chelsea head-to-head"]


**Prediction prompt** (replace `{team_a}`, `{team_b}`, `{features}`):
You are an objective football analyst. Given the structured summary below, answer in strict JSON with fields:
{"winner": "<Team A|Team B|Draw>", "probability": <0-100 number>, "reasoning": "<one-sentence reasoning>", "top_signals": ["signal1","signal2","signal3"]}

Data:
Team A: {team_a}
Team B: {team_b}
Extracted features JSON:
{features}

Rules:

Output only a single JSON object.

Probability is a number between 0 and 100 that reflects confidence.

Provide three concise top_signals supporting the decision.


**Parsing guidance**: try `json.loads()` first; if that fails, extract the first balanced JSON substring and parse.

---

# 7. Logging, persistence & reproducibility rules

- Save run logs to `runs/run_<unix_ts>.json` with keys: `teams`, `timestamp`, `queries`, `raw_tavily` (full provider JSON), `features`, `prompts_used`, `llm_responses`, `parsed_prediction`.  
- Do not commit API keys or models. `.env` should contain `TAVILY_API_KEY` and be in `.gitignore`.  
- Log provider quota warnings and raw error bodies for debugging.

---

# 8. Tests & CI guidance

- Implement pytest tests for the Tavily adapter and LLM wrapper as described.  
- In README include recommended manual CI steps: create env, run pytest, run a smoke demo (with models in place).  
- Do not add CI YAML automatically for MVP unless instructed.

---

# 9. Performance & safety heuristics

- Rate-limit Tavily calls and implement short per-request delay; log quota errors.  
- Cache Tavily responses per query (in-memory or simple file cache) to preserve credits.  
- Prefer Tavily Extract results over local Playwright scraping — call Extract only when snippets/extracted content lack signals.  
- Keep LLM context short: pass compact features rather than long raw text when possible.

---

# 10. Documentation links (agent must consult these)

The coding LLM must consult official docs for exact endpoints, headers, parameter names, and example responses:

- Tavily Python SDK Quickstart & Search API docs — installation, client usage, Search and Extract endpoints, and Extract params. (Docs show `pip install tavily-python` and client instantiation examples). :contentReference[oaicite:0]{index=0}  
- Tavily GitHub (Python wrapper repository) for reference & examples. :contentReference[oaicite:1]{index=1}  
- LangChain Tavily integration docs (optional reference; we are using Tavily as a direct client in Option B). :contentReference[oaicite:2]{index=2}  
- `llama.cpp` GitHub & `llama-cpp-python` docs (local LLM integration).  
- Streamlit docs (UI).  

(Agent must open the Tavily docs to copy exact headers, endpoint paths, and request examples during implementation.)

---

# 11. LLM action items — prioritized tasks (exact order)

**Priority 1 — core MVP**
1. Initialize git repo with the exact layout above.  
2. Add `env.yml` and `README.md` (include `pip install tavily-python` instruction and `TAVILY_API_KEY` usage).  
3. Implement `backend/search_adapter/base.py` and `backend/search_adapter/tavily.py` (with retries, normalization, and credit-aware options). Add unit tests mocking Tavily.  
4. Implement `backend/llm/llm_client.py` robust wrapper.  
5. Implement `backend/parse/parser.py` using Tavily content & Extract fallback.  
6. Implement `backend/runner.py` orchestration and run-logging.  
7. Implement `app.py` Streamlit UI and error handling.  
8. Run tests locally and produce `runs/demo_<ts>.json` for a sample match.  

**Priority 2 — polish**
9. Add simple file-based caching and local per-query rate limiter.  
10. Improve parsing heuristics and add parser unit tests.  

**Priority 3 — optional future work**
12. Add a small FastAPI wrapper.  
13. Add hosted LLM fallback (Hugging Face) configurable via env var.  
14. Add a CI skeleton if requested.

---

# 12. Validation checklist for the human reviewer

- Repo exists with exact layout and filenames replaced where necessary (`tavily.py` instead of `serper.py`).  
- `env.yml` creates the conda env and `pip install tavily-python` is included.  
- `app.py` runs (`streamlit run app.py`) and runner reports friendly errors if model absent.  
- A demo run (with models in `models/`) produces `runs/<ts>.json` with expected schema.  
- Tests for adapter and llm_client run and pass (mock-based).

---

# 13. Final notes for the coding LLM (strict guidance)

- **NO CODE IN THIS PLAN:** Implement files described above; do not output code in responses to this plan.  
- **Strict filenames:** Use the exact file names and structure above — `tavily.py` replaces `serper.py`.  
- **Prompts are authoritative:** Use the provided prompt templates verbatim unless a clear failure appears. If the LLM returns invalid JSON, log the failure and perform one corrective retry.  
- **Robust parsing required:** Try `json.loads()` first; else extract a balanced JSON substring.  
- **Log everything:** Each run must be fully persisted to `runs/` for reproducibility and debugging.  
- **Tavily package to install:** `tavily-python` (agent must add `pip install tavily-python` to env instructions). :contentReference[oaicite:3]{index=3}

