# 🧾 InvoiceAI — Financial Document RAG Assistant

<p align="left">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white" alt="Flask">
  </a>
  <a href="https://ai.google.dev/">
    <img src="https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white" alt="Gemini">
  </a>
  <a href="https://python.langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat&logo=chainlink&logoColor=white" alt="LangChain">
  </a>
  <a href="https://faiss.ai/">
    <img src="https://img.shields.io/badge/FAISS-Vector_DB-00979D?style=flat" alt="FAISS">
  </a>
  <a href="https://pypi.org/project/pypdf/">
    <img src="https://img.shields.io/badge/PyPDF-PDF_Processing-CC342D?style=flat" alt="PyPDF">
  </a>
</p>

> **A Retrieval-Augmented Generation system that answers questions about invoices — grounded strictly in the document, with citations, structured data extraction, and a real evaluation harness.**

Upload one or more invoice PDFs and ask questions in plain language. Every answer is traceable back to the exact page and chunk it came from — no hallucination, no guessing.

---

## 🚀 Overview

Most "ask your PDF" demos stop at a single document and a single free-text answer. InvoiceAI goes further:

- **Multiple invoices in one session** — ask cross-document questions, not just single-PDF Q&A
- **Page-level citations** — every answer names the source file, page number, and a relevance score
- **Structured field extraction** — a separate JSON-mode call pulls vendor, totals, dates, and invoice numbers into clean structured data, not just sentences
- **A real evaluation harness** — measures retrieval accuracy and answer accuracy *separately*, plus latency, instead of one vague "it works" check
- **Honest engineering** — relevance scores are labeled as retrieval heuristics, not fabricated confidence percentages; known limitations are documented, not hidden

---

## 🧠 RAG Pipeline

```text
PDF Upload(s)
     │
     ▼
Page-Level Text Extraction (PyPDFLoader)
     │
     ▼
Metadata Tagging (source file + page number)
     │
     ▼
Chunking (500 chars, 100 overlap)
     │
     ▼
Embeddings (Gemini Embedding API)
     │
     ▼
FAISS Vector Index (merge-able across multiple PDFs)
     │
     ▼
User Query
     │
     ▼
Similarity Search → Top-K Chunks + Relevance Scores
     │
     ▼
Anti-Hallucination Prompt (context + question)
     │
     ▼
Gemini LLM
     │
     ▼
Answer + Page Citations + Excerpts
```

A second, independent path handles structured extraction:

```text
Full Document Text → JSON-Schema Prompt → Gemini → Parsed JSON Fields
```

---

## ✨ Features

| Feature | What it does |
|---|---|
| 📄 Multi-PDF upload | Merge multiple invoices into one searchable session via FAISS index merging |
| 🎯 Grounded Q&A | Answers come only from retrieved document context — returns "Not found in document." otherwise |
| 📌 Page citations | Every answer shows which file, which page, and a relevance score per supporting chunk |
| 🧩 Structured extraction | One-click JSON extraction of vendor, client, invoice number, dates, total, currency |
| 🌗 Dark/light mode | Toggle with persisted preference, defaults to dark, with an animated background |
| 🗑️ Session reset | One button clears the uploaded documents, FAISS index, and Q&A history |
| 💾 Persistent chat history | Saved in browser localStorage — survives a page refresh |
| 📊 Evaluation harness | Standalone script measuring retrieval hit rate, answer accuracy, and latency |
| 🛡️ Rate limiting | Per-IP request limits protect the shared API key from being exhausted by one visitor |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Web Framework | Flask |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| LLM | Gemini 2.5 Flash (Google Generative AI) |
| Embeddings | Gemini Embedding API |
| PDF Processing | PyPDF |
| Frontend | Vanilla HTML/CSS/JS |
| Rate Limiting | Flask-Limiter |
| Environment Config | python-dotenv |

No databases, no Docker, no containerization — FAISS runs in-memory per session, intentionally kept simple for a single-instance deployment.

---

## 📂 Project Structure

```text
invoice-ai/
│
├── app.py                  # Flask routes: upload, ask, extract, reset
├── evaluation.py           # Standalone RAG evaluation harness
│
├── rag/
│   ├── ingestion.py        # Load, chunk, embed, multi-PDF FAISS merging
│   ├── retrieval.py        # Prompt engineering, retrieval, citations
│   └── extraction.py       # Structured JSON field extraction
│
├── templates/
│   └── index.html          # Single-page app structure
│
├── static/
│   ├── css/
│   │   ├── base.css        # Tokens, layout, header, footer
│   │   ├── upload.css       # Upload zone, progress, stat badges
│   │   ├── qa.css          # Query input, answer cards, history
│   │   ├── extract.css     # Structured field cards, citations
│   │   └── theme.css       # Dark mode + animated background
│   └── js/
│       ├── utils.js        # Shared state, theme toggle
│       ├── upload.js       # Multi-file drag-and-drop upload
│       ├── qa.js           # Ask, render answers + citations, history
│       ├── extract.js      # Structured field rendering
│       └── reset.js        # Session + history clearing
│
├── uploads/                 # Uploaded PDFs (created automatically)
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/NUBAID-UDDIN/Financial-Document-RAG-Assistant
cd invoice-ai
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_api_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/).

### 5. Run it

```bash
python app.py
```

Visit `http://127.0.0.1:5000`.

---

## Evaluation: how to calculate honest metrics

Do not quote a metric unless you can rerun it using the exact evaluation set. The older committed test set contained unverified placeholder values and special handling that incorrectly counted unanswerable questions as retrieval hits. The updated evaluator requires ground truth to be manually checked against the exact invoice PDF.

### 1. Create a test-case template

```powershell
python evaluation.py --make-template eval_cases.json
```

Open the invoice PDF and fill in each `accepted_answers` and `evidence_terms` array. For a field that is genuinely absent, set `answerable` to `false` only after checking every page. Keep the out-of-document control question to test whether the assistant abstains.

Example **schema only** (the values below are illustrative, not a result from this repository):

```json
{
  "test_cases": [
    {
      "id": "invoice_number",
      "question": "What is the invoice number?",
      "answerable": true,
      "accepted_answers": ["COPY_REAL_VALUE_FROM_PDF"],
      "evidence_terms": ["COPY_REAL_VALUE_FROM_PDF"],
      "match_mode": "text"
    },
    {
      "id": "amount",
      "question": "What is the total amount due?",
      "answerable": true,
      "accepted_answers": ["COPY_REAL_AMOUNT_FROM_PDF"],
      "evidence_terms": ["COPY_REAL_AMOUNT_FROM_PDF"],
      "match_mode": "number"
    },
    {
      "id": "outside_question",
      "question": "What is the capital of France?",
      "answerable": false,
      "accepted_answers": [],
      "evidence_terms": [],
      "match_mode": "text"
    }
  ]
}
```

Do not leave `COPY_REAL...` placeholders in the file. The evaluator validates the case file and refuses to calculate if the ground truth is missing.

### 2. Run evaluation

```powershell
python evaluation.py path\to\invoice.pdf --cases eval_cases.json --output eval_report.json --k 4
```

The script reports:

- **Retrieval hit rate:** among answerable questions only, how often annotated evidence terms appear in at least one retrieved chunk. Unanswerable questions are excluded from this denominator.
- **Answer accuracy:** the fraction of all labeled questions where the final answer matches a verified accepted answer, or the system abstains on a labeled unanswerable question.
- **Abstention accuracy:** among unanswerable questions, how often the assistant abstains.
- **Latency:** wall-clock time for retrieval and generation, including API/network time.

The JSON report records each question, generated answer, retrieval result, latency, errors, and the metric definitions. Evidence-term matching is a transparent proxy, not a full human semantic relevance assessment. Small test sets have high uncertainty.

---

## 🧩 Core Components

| Component | Purpose |
|---|---|
| `PyPDFLoader` | Extracts page-level text and attaches source metadata |
| `RecursiveCharacterTextSplitter` | Splits text into overlapping chunks, preserving metadata |
| `GoogleGenerativeAIEmbeddings` | Converts text chunks into vectors |
| `FAISS` | Stores vectors, supports merging multiple documents into one index |
| `retrieve_chunks()` | Similarity search + relevance scoring |
| `answer_query()` | Builds the grounded prompt, calls Gemini, attaches citations |
| `extract_invoice_fields()` | Separate structured-output call returning strict JSON |
| `evaluation.py` | Measures retrieval and answer accuracy independently |

---

## ⚠️ Error Handling

The application handles:

- ❌ Missing or invalid API key
- ❌ Missing or non-PDF file uploads
- ❌ Empty user queries
- ❌ No relevant context retrieved
- ❌ Malformed JSON from structured extraction (parsed defensively, fails gracefully)
- ❌ Gemini API rate limits (429) — mitigated with per-IP rate limiting so one visitor can't exhaust the shared quota for everyone

---

## 🎯 Key Engineering Decisions (and why)

- **Relevance score, not "confidence"** — the UI shows a retrieval relevance percentage derived from FAISS distance. It is explicitly *not* presented as a calibrated probability, because it isn't one. Honesty about what a metric actually measures matters more than a more impressive-looking number.
- **Two separate LLM call patterns** — free-text grounded Q&A (RAG) and structured JSON extraction are implemented as two distinct code paths, because they're solving two different problems with two different prompting strategies.
- **In-memory session state** — no database is used; each visitor's FAISS index and document state live in a server-side dictionary. This is a deliberate simplicity tradeoff for a single-instance portfolio deployment, not a production-scale design, and is documented here rather than hidden.
- **Evaluation measures two failure modes separately** — a wrong final answer can mean either "retrieval found the wrong chunk" or "retrieval was right but the LLM misread it." Measuring `retrieval_hit` and `answer_correct` independently makes it possible to tell which failure actually occurred.

---

## 🔒 Known Limitations

- No persistent database — FAISS indexes reset on server restart
- No authentication or multi-tenant isolation beyond per-session cookies
- Relevance scoring is a heuristic, not a calibrated confidence metric
- Evaluation test cases must be manually written per document — there is no automated ground-truth generation

---

## 🔮 Future Enhancements

- OCR support for scanned (non-text) invoices
- Multi-invoice comparison ("which invoice has the highest amount?")
- Automated test-case generation from document content
- Persistent storage layer for session state across restarts
- Bring-your-own-API-key support for public deployments

---

## 📚 What This Project Demonstrates

- Retrieval-Augmented Generation (RAG) architecture end-to-end
- Prompt engineering for hallucination prevention with a defined fallback behavior
- Vector embeddings and FAISS similarity search, including incremental index merging
- Structured output / JSON-mode prompting as a distinct pattern from free-text generation
- RAG evaluation methodology — separating retrieval accuracy from generation accuracy
- Flask API design with session-scoped state and per-IP rate limiting
- Frontend engineering without a framework — modular CSS/JS, dark mode via CSS variables, localStorage-backed state

---
