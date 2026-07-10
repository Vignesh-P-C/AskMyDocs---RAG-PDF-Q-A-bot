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

# AskMyDocs
### Stop skimming. Start asking.
#### | Turn any PDF into a conversation.

Upload a PDF, ask a question, get an answer grounded in the document's actual content — powered by Retrieval-Augmented Generation (RAG).

## How it works

1. **Document loading** — `PyPDFLoader` (LangChain) extracts text from the uploaded PDF
2. **Text splitting** — `RecursiveCharacterTextSplitter` breaks the document into overlapping chunks
3. **Embedding** — an IBM watsonx embedding model converts each chunk into a vector
4. **Vector storage** — Chroma stores the embeddings for fast similarity search
5. **Retrieval** — a similarity-search retriever pulls the most relevant chunks for a given question
6. **Generation** — an IBM watsonx LLM generates an answer using the retrieved context, via LangChain's `RetrievalQA` chain
7. **Interface** — Gradio provides the web UI

## Demo

![AskMyDocs demopic](assets/demopic.png)
![AskMyDocs demogif](assets/demogif.gif)

## Tech stack

- Python 3.11
- [LangChain](https://www.langchain.com/) + `langchain-ibm` + `langchain-community`
- [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) (LLM + embeddings)
- [Chroma](https://www.trychroma.com/) vector database
- [Gradio](https://www.gradio.app/) front-end

## Running it yourself

```bash
python3.11 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
python3.11 app.py
```

Then open the local URL Gradio prints, upload a PDF, and ask a question.

**Note on credentials:** this project requires an IBM watsonx.ai project ID and API key. Set these via environment variables rather than hardcoding them — see `app.py` for where credentials are configured.

## Files

| File | Purpose |
|---|---|
| `app.py` | Main application — RAG pipeline + Gradio interface |
| `test_embedding.py` | Standalone script demonstrating the embedding model in isolation |
| `requirements.txt` | Python dependencies |

## Acknowledgment

Originally built as the capstone project for IBM's *Generative AI Applications with RAG and LangChain* course (Coursera), later refactored and rebranded as AskMyDocs.
