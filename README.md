#  Legal Judgement Summarizer

Upload any legal PDF and get instant structured summaries using RAG.

## What it does
- Upload a legal PDF document
- Ask any question about it
- Get structured output: Summary, Key Points, Final Verdict

## Tech Stack
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Document embeddings
- **Groq (LLaMA 3.3-70b)** — LLM summarization
- **PyPDF** — PDF text extraction
- **Streamlit** — Frontend UI

##  Setup
```bash
pip install -r requirements.txt
```
Add your Groq API key to `.env`:
