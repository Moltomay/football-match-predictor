# Football Match Result Predictor

A Streamlit-based prototype that predicts football match outcomes using a local LLM and Tavily for real-time search and extraction.

## Features
- Automated search query generation for match data.
- Search and content extraction via Tavily.
- Heuristic-based feature parsing from search results.
- Outcome prediction with confidence score and reasoning using a local LLM.
- Detailed run logs for transparency.

## Prerequisites
- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (recommended)
- Tavily API Key
- GGUF Quantized LLM (e.g., Mistral-7B, Llama-3-8B)

## Installation

1. **Create Conda Environment**:
   ```bash
   conda env create -f env.yml
   conda activate footy-predictor
   ```

2. **Install Playwright Browsers** (optional, for fallback):
   ```bash
   python -m playwright install
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory:
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   MODEL_PATH=path/to/your/model.gguf
   ```

4. **Prepare Models**:
   Place your GGUF model in the `models/` directory or update `MODEL_PATH` in `.env`.

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## Testing
Run unit tests:
```bash
pytest
```
