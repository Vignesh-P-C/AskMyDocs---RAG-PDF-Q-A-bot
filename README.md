---
title: AskMyDoc
emoji: 🐨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: '3.11'
app_file: app.py
pinned: false
license: mit
short_description: Stop skimming. Start asking. Turn any PDF into a convo.
---

# AskMyDocs — Talk to Your PDFs

[![Live Demo](https://img.shields.io/badge/LIVE_DEMO-HUGGING_FACE-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](PASTE_SPACE_URL_HERE)
[![Source](https://img.shields.io/badge/SOURCE-GITHUB-black?style=for-the-badge&logo=github)](PASTE_GITHUB_REPO_URL_HERE)
![Model](https://img.shields.io/badge/MODEL-GPT--OSS--120B-orange?style=for-the-badge&logo=huggingface)
![Framework](https://img.shields.io/badge/FRAMEWORK-LANGCHAIN-1C3C3C?style=for-the-badge)
![Vector Store](https://img.shields.io/badge/VECTOR_STORE-CHROMA-6A0DAD?style=for-the-badge)
![Hardware](https://img.shields.io/badge/HARDWARE-ZEROGPU-blue?style=for-the-badge)

> Stop skimming. Start asking. Upload a PDF, ask it anything, and get answers grounded in the document's actual content — powered by Retrieval-Augmented Generation (RAG), running end-to-end on a free Hugging Face Space.

| | Link |
|---|---|
| 🚀 Live demo | [Try it on Hugging Face](https://huggingface.co/spaces/V1gnesh/AskMyDoc) |
| 💻 Source | [GitHub repo](https://github.com/Vignesh-P-C/AskMyDocs---RAG-PDF-Q-A-bot) |

> ⚠️ This Space runs on Hugging Face's free ZeroGPU tier. If it's been idle a while, the first request after waking may take a little longer while the Space spins back up.

---

## 📸 Demo

![AskMyDocs demopic](assets/demopic.png)
![AskMyDocs demogif](assets/demogif.gif)

---

## Overview

Upload a PDF, ask a question in plain English, and get an answer pulled straight from the document — not a generic response, not a hallucination, an answer grounded in what's actually on the page.

Originally built as an IBM watsonx.ai capstone project, AskMyDocs was later migrated entirely off IBM's stack onto Hugging Face Inference Providers after the original integration was retired. See [Project history](#project-history) below for how that transition went.

---

## How it works

1. **Upload** — a PDF is uploaded once; `PyPDFLoader` (LangChain) extracts its text.
2. **Split** — `RecursiveCharacterTextSplitter` breaks the text into overlapping chunks.
3. **Embed** — a local `sentence-transformers` model converts each chunk into a vector, entirely on CPU — no API call, no inference credit spent.
4. **Store** — Chroma indexes the vectors for fast similarity search. This happens **once per document**, not once per question.
5. **Ask** — for each question, a retriever pulls the most relevant chunks.
6. **Generate** — a LangChain LCEL chain feeds the retrieved context and question to `openai/gpt-oss-120b` (via Hugging Face Inference Providers, wrapped for chat mode) and parses out the answer.
7. **Chat** — Gradio's `Blocks` UI keeps the whole conversation, not just one answer at a time.

---

## System Architecture

```
PDF upload
     │
     ▼
PyPDFLoader → RecursiveCharacterTextSplitter
     │
     ▼
sentence-transformers  (local, CPU-only embeddings)
     │
     ▼
Chroma vector store        ← indexed once, cached for the session
     │
     ▼  similarity search
     │
Question  ──────────────────┤
     │                       ▼
     │              LangChain LCEL chain
     │        (context | prompt | ChatHuggingFace | parser)
     │                       │
     │                       ▼
     │        Hugging Face Inference Providers
     │              (openai/gpt-oss-120b)
     │                       │
     ▼                       ▼
        Answer, shown in the Gradio chat
```

---

## Features

- One-time PDF indexing per session — follow-up questions skip re-embedding and answer noticeably faster
- Real multi-turn conversation, not a single question/answer box
- Local CPU embeddings — the small free Hugging Face Inference credit budget is spent only on answer generation, not on every chunk
- Runs on Hugging Face Spaces' free ZeroGPU tier, with no local GPU required
- Custom themed Gradio `Blocks` interface

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | `openai/gpt-oss-120b` via Hugging Face Inference Providers |
| Orchestration | LangChain (LCEL) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | [Chroma](https://www.trychroma.com/) |
| UI | [Gradio](https://www.gradio.app/) (`Blocks`) |
| Hosting | Hugging Face Spaces (ZeroGPU) |

---

## Key Engineering Decisions

**Local embeddings over API calls** — Hugging Face's free Inference credit is small. Computing embeddings for every PDF chunk via a remote call would burn through that budget fast on cheap, high-volume calls. Running the embedding model locally on CPU reserves the entire free budget for the one call that actually needs a capable model: answer generation.

**Index once per session, not once per question** — the retriever is built when a PDF is uploaded and cached in Gradio's session state, rather than being rebuilt from scratch on every single question.

**`ChatHuggingFace` wrapper, not a raw endpoint** — Hugging Face Inference Providers serve `gpt-oss-120b` in conversational mode only; a raw completion-style call fails with a task-mismatch error.

**CPU-forced embeddings on a ZeroGPU Space** — on ZeroGPU hardware, `torch.cuda.is_available()` reports `True` even outside a GPU-allocated function, which makes `sentence-transformers` try (and fail) to initialize CUDA. Explicitly forcing `device="cpu"` on the embedding model sidesteps this entirely — consistent with the design decision above to keep embeddings off any GPU allocation anyway.

---

## Project History

This started as the capstone for IBM's *Generative AI Applications with RAG and LangChain* course (Coursera), originally built on `langchain-ibm` and IBM watsonx.ai for both the LLM and embeddings.

It was later rebranded as AskMyDocs and migrated fully onto Hugging Face: watsonx was replaced with Hugging Face Inference Providers for generation and local `sentence-transformers` for embeddings, the LangChain integration was upgraded (which meant following several import paths that had moved, and rewriting the retrieval chain from the now-deprecated `RetrievalQA` to an LCEL pipe chain), and the UI was rebuilt from a bare `gr.Interface` into a themed `gr.Blocks` app with session-cached retrieval and real conversation history.

---

## Run Locally

```bash
py -3.12 -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# or: source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install gradio spaces        # not bundled in requirements.txt — install separately

python app.py
```

Then open the local URL Gradio prints, upload a PDF, and ask a question.

**Note on credentials:** this project requires a Hugging Face access token. Create a **Fine-grained** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) using the **"Inference"** preset (a plain "Read" token is not sufficient — it will 403). Set it as `HUGGINGFACEHUB_API_TOKEN` in a local `.env` file for testing, or as a Space secret (Settings → Variables and secrets) when deploying — see `.env.example`.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Main application — RAG pipeline + Gradio `Blocks` interface |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for the required `HUGGINGFACEHUB_API_TOKEN` |

---

## Roadmap

- [ ] Decoupled version: React frontend (Vercel) + FastAPI backend, as a separate project
- [ ] Multi-document Q&A in a single session
- [ ] Source-chunk citations alongside each answer
- [ ] Persist conversation history beyond a single session

---

## Contact

**Vignesh P C** — [GitHub](https://github.com/Vignesh-P-C) · [LinkedIn](https://www.linkedin.com/in/vignesh-p-c/)

<div align="center">
  <sub>Built with LangChain · Hugging Face Inference Providers · Chroma · Gradio</sub>
</div>