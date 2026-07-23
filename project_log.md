# AskMyDocs — Full Project Log & Handoff Document

**Purpose of this document:** This is a complete record of a project that started as an IBM Coursera GenAI course final assignment and evolved into a deployed, rebranded personal portfolio project called **AskMyDocs**. It is written so that a new conversation (with Claude or anyone else) can pick up exactly where things left off, without needing the original chat history.

**Owner:** Vignesh
**Original context:** IBM "Generative AI Applications with RAG and LangChain" course on Coursera (final project, Module 3)
**Current context:** Deployed personal project, GitHub + Hugging Face Spaces

---

## Table of Contents

1. [Project Origin & Requirements](#1-project-origin--requirements)
2. [Architecture Overview](#2-architecture-overview)
3. [Full Chronological Development Log](#3-full-chronological-development-log)
4. [Confirmed Technical Facts (Critical Reference Info)](#4-confirmed-technical-facts-critical-reference-info)
5. [Current State of All Files](#5-current-state-of-all-files)
6. [Current Deployment Status](#6-current-deployment-status)
7. [Known Open Issues / Unresolved Items](#7-known-open-issues--unresolved-items)
8. [Recommended Next Steps (Prioritized)](#8-recommended-next-steps-prioritized)
9. [Quick-Reference Command Cheat Sheet](#9-quick-reference-command-cheat-sheet)

---

## 1. Project Origin & Requirements

### 1.1 The course assignment

The final project for IBM's GenAI Coursera course (Module 3: "Create a QA Bot to Read Your Document") required building a question-answering bot that:

- Loads a PDF document
- Splits it into chunks
- Embeds the chunks using an embedding model
- Stores embeddings in a vector database
- Retrieves relevant chunks based on a user's query
- Uses an LLM to generate an answer via Retrieval-Augmented Generation (RAG)
- Exposes all of this through a Gradio web interface

The official assignment provided a **fill-in-the-blank code template** for a file called `qabot.py`, built around IBM's **watsonx.ai** platform (via the `ibm-watsonx-ai` SDK and `langchain-ibm`), running inside IBM's **Skills Network (SN) Labs** cloud IDE (a Theia-based browser IDE).

### 1.2 The official required functions (from the graded template)

These six functions were explicitly required, with these exact names:

| Function | Required behavior |
|---|---|
| `get_llm()` | Initialize a `WatsonxLLM` instance, temperature=0.5, max_new_tokens=256 |
| `document_loader(file)` | Load a PDF using `PyPDFLoader` from `langchain_community` |
| `text_splitter(data)` | Split text using `RecursiveCharacterTextSplitter`, chunk_size=1000 |
| `watsonx_embedding()` | Return a `WatsonxEmbeddings` instance from `langchain_ibm` |
| `vector_database(chunks)` | Embed chunks and store in `Chroma.from_documents()` |
| `retriever(file)` | Chain load → split → embed → `.as_retriever()` |
| `retriever_qa(file, query)` | Use `RetrievalQA.from_chain_type()` with `get_llm()` and `retriever()`, return `response['result']` |

Plus a Gradio interface with:
- `gr.File()` for PDF upload
- `gr.Textbox()` for the query input
- `gr.Textbox()` for the output
- `rag_application.launch(server_name=..., server_port=7860)`

### 1.3 Required deliverables for grading (peer-graded assignment)

Six screenshots, each with an exact required filename:

| Filename | Content required |
|---|---|
| `pdf_loader.png` | Code screenshot of `document_loader(file)` |
| `code_splitter.png` | Code screenshot of `text_splitter(data)` |
| `embedding.png` | Code **+ actual output** — must show the code used to embed a test sentence AND the resulting first 5 embedding numbers printed |
| `vectordb.png` | Code screenshot of `vector_database(chunks)` |
| `retriever.png` | Code screenshot of `retriever(file)` |
| `QA_bot.png` | **Webpage screenshot** of the running Gradio app, with a PDF uploaded and the exact query `What this paper is talking about?` typed in, with a visible answer in the Output box |

A separate quiz (Question 10, Question 11, etc.) accompanies the project:
- **Question 10** (2 points, upload type): required uploading a screenshot named `QA_bot.png` — later clarified this question actually wants **all task screenshots** uploaded together, not just the one file.
- **Question 11** (2 points, multiple select): "Which of the following components are used in the QA bot implementation to interact with the user?" — correct answers: **`gr.Textbox()`** and **`gr.File()`**. Incorrect: `gr.Slider()`, `gr.Image()` (not used anywhere in this project).

---

## 2. Architecture Overview

The bot implements a standard **Retrieval-Augmented Generation (RAG)** pipeline:

```
PDF upload
   │
   ▼
[1] document_loader()  →  PyPDFLoader extracts raw text from PDF
   │
   ▼
[2] text_splitter()    →  RecursiveCharacterTextSplitter breaks text into
   │                      overlapping chunks (chunk_size=1000, overlap=200)
   ▼
[3] watsonx_embedding() →  IBM watsonx embedding model converts each chunk
   │                       into a vector representation
   ▼
[4] vector_database()  →  Chroma stores all chunk vectors for fast
   │                      similarity search
   ▼
[5] retriever()        →  Wraps the vector store as a retriever object
   │                      (similarity search under the hood)
   ▼
[6] retriever_qa()     →  On a user query: retrieves the most relevant
   │                      chunks, stuffs them into a prompt, sends to
   │                      the watsonx LLM via LangChain's RetrievalQA
   │                      chain, returns the generated answer
   ▼
[7] Gradio interface   →  Displays PDF upload box, query textbox, and
                          output textbox in the browser
```

---

## 3. Full Chronological Development Log

### Phase 1 — Initial code scaffold

Built the first complete version of `qabot.py` from scratch, implementing all 6 required functions plus the Gradio interface and launch call, based on general knowledge of this well-known IBM lab pattern (before the official fill-in-the-blank template was shared).

Initial (later-corrected) choices:
- LLM: `mistralai/mixtral-8x7b-instruct-v01`
- Embedding: `ibm/slate-125m-english-rtrvr`
- Imports initially used `langchain.document_loaders` / `langchain.vectorstores` (later corrected to `langchain_community.*`)

### Phase 2 — SN Labs environment setup

The official lab instructions specified:
```bash
cd /home/project
pip install virtualenv
virtualenv my_env
source my_env/bin/activate

python3.11 -m pip install \
gradio==4.44.0 \
jinja2==3.1.4 \
pydantic==2.10.6 \
huggingface_hub==0.23.0 \
fastapi==0.112.2 \
starlette==0.37.2 \
uvicorn==0.30.6 \
ibm-watsonx-ai==1.1.2 \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11
```
This list was installed successfully. **Note:** this pinned list did NOT include `chromadb` or `pypdf`, which the code actually needs — this caused an error later (see Phase 6).

A separate, later-shared version of the official lab page gave a slightly different install list (for reference, in case of future conflicts):
```bash
python3.11 -m pip install \
gradio==4.44.0 \
ibm-watsonx-ai==1.1.2 \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11 \
chromadb==0.4.24 \
pypdf==4.3.1 \
pydantic==2.9.1 \
huggingface_hub==0.23.0
```
In practice, `pip install pypdf chromadb` (unpinned) was used instead, pulling whatever the latest compatible versions were at the time — this worked fine in practice.

### Phase 3 — File location mistake

The `qabot.py` file was accidentally created **inside the `.theia` config folder** instead of the project root, causing `python3.11 qabot.py: No such file or directory` when run from `/home/project`. Fixed by moving the file to sit directly under `PROJECT` (i.e., `/home/project/qabot.py`), as a sibling of `.theia` and `my_env`.

### Phase 4 — Browser access / proxy URL issues

Attempting to access the running Gradio app via `https://<lab-domain>:7860` directly failed with `ERR_CONNECTION_TIMED_OUT`, because SN Labs' cloud IDE doesn't expose raw ports directly — it requires either:
- The **Skills Network Toolbox** sidebar panel → "Launch Application" → enter port `7860`, OR
- Directly visiting the proxy URL pattern:
  ```
  https://<username>-7860.theianext-0-labs-prod-misc-tools-us-east-0.proxy.cognitiveclass.ai/
  ```
This proxy URL pattern is what ultimately worked and was used throughout the rest of the SN Labs work.

Also discovered: running `python3.11 qabot.py` a second time while a previous instance was still bound to port 7860 caused silent hangs. Fix pattern used repeatedly:
```bash
lsof -i :7860        # check what's using the port
kill -9 <PID>         # kill it if needed
python3.11 qabot.py   # restart cleanly
```

### Phase 5 — Model support errors (the biggest recurring issue)

This was the most iterative part of the whole process. IBM's watsonx SDK has a **client-side validation list** of models it generally considers valid, but the **actual live API** enforces a much more specific, restricted list of models the given SN Labs project (`project_id="skills-network"`) is actually entitled to use. These two lists disagreed multiple times, causing a chain of trial-and-error:

1. **`mistralai/mixtral-8x7b-instruct-v01`** (initial LLM choice) → rejected by the SDK's own client-side check with error:
   ```
   Model 'mistralai/mixtral-8x7b-instruct-v01' is not supported for this environment.
   Supported models: ['ibm/granite-4-h-small', 'ibm/granite-8b-code-instruct',
   'ibm/granite-guardian-3-8b', 'meta-llama/llama-3-3-70b-instruct',
   'meta-llama/llama-4-maverick-17b-128e-instruct-fp8', 'mistralai/mistral-medium-2505',
   'mistralai/mistral-small-3-1-24b-instruct-2503', 'openai/gpt-oss-120b']
   ```
2. Switched to **`meta-llama/llama-3-3-70b-instruct`** (from that list) → passed the client-side check, but failed later at actual runtime with a **live HTTP 422 error** from the real watsonx API:
   ```
   Status code: 422: Model meta-llama/llama-3-3-70b-instruct is not supported.
   Supported Models: ibm/granite-embedding-278m-multilingual,
   mistralai/mistral-small-3-1-24b-instruct-2503, mistralai/mistral-medium-2505,
   meta-llama/llama-4-maverick-17b-128e-instruct-fp8, ibm/granite-4-h-small
   ```
3. This second, shorter list — confirmed identically by **two separate live API 422 errors** (once for the embedding model, once for the LLM) — is the **real, authoritative list** of models this specific SN Labs project is entitled to use. This is more restrictive than what the SDK's local validation allows.

**⭐ CONFIRMED WORKING MODELS for the `project_id="skills-network"` SN Labs sandbox (verified via live API calls, not just docs):**
| Model ID | Type |
|---|---|
| `ibm/granite-embedding-278m-multilingual` | Embedding |
| `mistralai/mistral-small-3-1-24b-instruct-2503` | LLM (fast, used as final choice) |
| `mistralai/mistral-medium-2505` | LLM |
| `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | LLM |
| `ibm/granite-4-h-small` | LLM (matches the *official* assignment template's hardcoded choice, but was slower/seemed to hang once on a larger PDF) |

Final LLM choice settled on: **`mistralai/mistral-small-3-1-24b-instruct-2503`** (fast, reliable, gave good answers). Briefly switched to `ibm/granite-4-h-small` to match the official template exactly, then reverted back to `mistral-small-3-1-24b-instruct-2503` after it appeared slow/unresponsive on a larger PDF (~900KB) — though this may have just needed more patience rather than being truly broken, since a later test with the same model DID succeed on that same large PDF after waiting.

### Phase 6 — Missing packages

Running the app threw:
```
ModuleNotFoundError: No module named 'pypdf'
```
Fixed with:
```bash
python3.11 -m pip install pypdf chromadb
```
(These weren't in the originally-given pinned install list — see Phase 2 note.)

### Phase 7 — Wrong import for embedding parameters

Initial code used:
```python
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
...
embed_params = {
    EmbeddingTypes.EMBED_TEXT: { ... }
}
```
This threw `AttributeError: EMBED_TEXT` — `EmbeddingTypes` is meant for referencing embedding *model IDs*, not parameter keys. Fixed by switching to the correct class:
```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
...
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
    EmbedParams.RETURN_OPTIONS: {"input_text": True},
}
```

### Phase 8 — Embedding model also not entitled

Initial embedding model `ibm/slate-125m-english-rtrvr` (the model literally named in the reading materials) failed with a live 422 error — not on the entitled list (see Phase 5's confirmed list). Switched to `ibm/granite-embedding-278m-multilingual`, which worked.

### Phase 9 — First successful end-to-end run

After all the above fixes, the app successfully:
- Loaded a PDF (`Updated_Dataset_Description_Guide.pdf`)
- Answered "explain the contents of this file in points..." with an accurate, relevant, detailed answer

This confirmed the entire pipeline (loader → splitter → embedder → vector store → retriever → LLM → Gradio) was functioning correctly end-to-end.

### Phase 10 — Verification against the official fill-in-the-blank template

Once the official graded template was shared, a full line-by-line comparison was done. Result: the code's structure, function names, and logic matched the template's requirements exactly. Two voluntary alignment tweaks were made:
1. Switched LLM to `ibm/granite-4-h-small` to match the template's literal hardcoded model choice (later reverted per Phase 5).
2. Updated the Gradio `description=` string to match the template's exact given wording: *"Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."*

Also noted (but not corrected, since it's a template typo, not a real requirement): the official template's `watsonx_embedding()` code block has a stray trailing comma right after the `embed_params` dict's closing brace (`},`), which would technically make it a tuple, not a dict, if copied literally. The actual code here correctly does NOT have that stray comma.

### Phase 11 — Screenshot requirements clarified

Went back and forth on exactly what Question 10's upload wanted. Final understanding: **all 6 screenshots** need to be captured and uploaded together (not spread across separate questions as initially assumed). Full breakdown of what each screenshot needs (5 are code screenshots, 1 — `QA_bot.png` — is a webpage/browser screenshot) is documented in section 1.3 above.

### Phase 12 — `test_embedding.py` created

The `embedding.png` requirement is unique: it needs the `watsonx_embedding()` code **plus actual printed output** (first 5 embedding numbers for a test sentence), not just the function sitting unused. A standalone script was created for this:

```python
# test_embedding.py
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

def watsonx_embedding():
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
        EmbedParams.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

sentence = "How are you?"
embedding_model = watsonx_embedding()
result = embedding_model.embed_query(sentence)

print(f"Sentence: {sentence}")
print(f"First 5 embedding numbers: {result[:5]}")
```

Run via `python3.11 test_embedding.py`, produced:
```
Sentence: How are you?
First 5 embedding numbers: [-0.04264726862311363, 0.00730109354481101, -0.0077635785564780235, 0.028883717954158783, 0.03201628103852272]
```
Screenshotted alongside the code — satisfied the `embedding.png` requirement. **Confirmed done.**

### Phase 13 — Quiz answered

- **Question 10** (upload `QA_bot.png` / all screenshots): a working screenshot was confirmed — Gradio app running, a PDF about "Artificial Intelligence (AI) in Human Resource Management" uploaded, query typed exactly as `What this paper is talking about?`, and a correct, coherent answer generated. **Confirmed done** for this specific screenshot.
- **Question 11** (multiple select on Gradio components used): answered `gr.Textbox()` and `gr.File()` — confirmed correct based on the actual code.

### Phase 14 — GitHub repository setup (first pass, before rebrand)

Created a bundle of files for a GitHub repo under the original working name "qabot": `qabot.py`, `test_embedding.py`, `requirements.txt`, `.gitignore`, `README.md` (with setup instructions, tech stack, architecture explanation).

### Phase 15 — Rebrand to "AskMyDocs"

Brainstormed project names; landed on **AskMyDocs**. Tagline chosen: **"Your documents, answered."** (Later, the user's own README also included: *"Stop skimming. Start asking. Turn any PDF into a conversation."* as a subtitle/short_description.)

Restructured into:
```
askmydocs/
├── app.py                  (renamed from qabot.py — HF Spaces expects app.py)
├── requirements.txt
├── README.md
├── test_embedding.py
├── .gitignore
└── assets/
    └── (demo screenshot/gif)
```
Updated `app.py`'s Gradio `title=` to `"AskMyDocs — Ask Questions About Your PDF"`.

### Phase 16 — Deployment platform decision

**Vercel was considered and rejected** — reasoning: Vercel is built for serverless functions/static sites with short request cycles; this app is a long-running, stateful Python server with heavy dependencies (`chromadb`, `langchain`, `ibm-watsonx-ai`) that don't fit serverless cold-start constraints well.

**Chosen instead: Hugging Face Spaces** (purpose-built for Gradio apps, free tier available). Alternatives mentioned for reference: Render, Railway, Fly.io.

**Critical blocker flagged at this stage (still unresolved as of this document):** the code uses `project_id = "skills-network"`, a shortcut that **only works inside IBM's Skills Network cloud labs**. It will fail authentication anywhere else, including on Hugging Face. This needs real IBM Cloud watsonx credentials before the public deployment can actually answer questions. **See Section 7 for full detail — this is the most important open item.**

### Phase 17 — GitHub push (local machine)

On the local Windows machine (`D:\Vignesh\Projects\AskMyDocs - PDF QA_bot`):
```powershell
git init
git add .
git commit -m "Initial commit: AskMyDocs RAG PDF Q&A bot"
git remote add origin https://github.com/Vignesh-P-C/AskMyDocs---RAG-PDF-Q-A-bot.git
git branch -M main
git push -u origin main
```
**Succeeded** — GitHub repo is live at:
`https://github.com/Vignesh-P-C/AskMyDocs---RAG-PDF-Q-A-bot`

### Phase 18 — Hugging Face Space creation & first push conflict

Created Space at `https://huggingface.co/spaces/V1gnesh/AskMyDoc` (note: named **"AskMyDoc"**, no trailing "s" — inconsistent with the GitHub repo name "AskMyDocs"; flagged as a cosmetic issue, not yet resolved).

```powershell
git remote add space https://huggingface.co/spaces/V1gnesh/AskMyDoc
git push space main
```
**Rejected** — `! [rejected] main -> main (fetch first)`. Cause: Hugging Face auto-initializes new Spaces with their own starter commit (a default README with special YAML metadata), creating a separate, unrelated git history from the local repo.

**Fix:**
```powershell
git pull space main --allow-unrelated-histories
```
This caused a **merge conflict in README.md** (both sides had one).

### Phase 19 — README YAML merge conflict resolution

Hugging Face Spaces require a specific YAML front-matter block at the very top of `README.md` to know how to build/run the app. The conflict was resolved by keeping this block first, then the project's normal README content below it:

```yaml
---
title: AskMyDoc
emoji: 🐨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Stop skimming. Start asking. Turn any PDF into a convo.
---
```

**Two issues flagged at this point (unclear if fixed — see Section 7):**
1. `python_version: '3.12'` — the app was built and tested entirely on **Python 3.11**; recommended changing to `'3.11'` to match, since the pinned dependency versions were validated against 3.11, not 3.12.
2. `title: AskMyDoc` vs. body heading `# AskMyDocs` — naming inconsistency, purely cosmetic, recommended picking one consistently.

After resolving:
```powershell
git add README.md
git commit -m "Merge Hugging Face Space metadata with project README"
git push space main
```

### Phase 20 — Binary file rejection → Git LFS setup

Push rejected again:
```
remote: Your push was rejected because it contains binary files.
remote: Please use https://huggingface.co/docs/hub/xet to store binary files.
Offending files: assets/demogif.gif, assets/demopic.png
```
Hugging Face requires binary files to go through Git LFS. Fix attempt #1:
```powershell
git lfs install
git lfs track "*.png"
git lfs track "*.gif"
git add .gitattributes
git add assets/demopic.png assets/demogif.gif
git commit -m "Track image assets with Git LFS"
git push space main
```
**Still rejected** — because the *original* commit (before LFS tracking existed) still had these files as raw binary blobs in history, and HF scans the entire pushed history, not just the latest commit.

**Actual fix — rewrite history:**
```powershell
git lfs migrate import --include="*.png,*.gif" --everything
git push --force space main
git push --force origin main
```
This rewrote all historical commits to use LFS pointers instead of raw binaries for these file types. **This finally succeeded** in getting past the binary-file rejection.

### Phase 21 — Dependency conflict cascade during HF build

Once the binary issue was resolved, the **build** itself started failing due to dependency conflicts, because Hugging Face Spaces auto-installs Gradio based on the README's `sdk_version` field (`6.20.0` — a much newer major version than the `4.44.0` the app was built for), and the project's own `requirements.txt` kept separately pinning older, conflicting versions of shared dependencies. This played out as a chain of conflicts, fixed one at a time:

1. **Conflict 1:** `requirements.txt` pinned `gradio==4.44.0`, clashing directly with the forced `gradio==6.20.0`. **Fix:** removed the `gradio==4.44.0` line from `requirements.txt` entirely (let HF manage Gradio's version).
2. **Conflict 2:** `huggingface_hub==0.23.0` pin clashed with Gradio 6.20.0's requirement of `huggingface-hub<2.0,>=1.2.0`. **Fix:** removed that pin too.
3. **Conflict 3:** `pydantic==2.10.6` pin clashed with Gradio 6.20.0's `mcp` extra, which needs `pydantic>=2.11.10`. **Fix:** removed that pin too.

**Recommendation given but not fully acted upon:** rather than continuing to strip pins one at a time as new conflicts surfaced, it was suggested to instead **lower `sdk_version` in the README to `4.44.0`** (matching what the app was actually built and tested against), which likely would have avoided this whole cascade in one move. The user chose to continue with pin-stripping instead, which did eventually work.

**Final `requirements.txt` after all three removals:**
```
ibm-watsonx-ai==1.1.2
langchain==0.2.11
langchain-community==0.2.10
langchain-ibm==0.1.11
chromadb
pypdf
```

### Phase 22 — Runtime error: `allow_flagging` removed in Gradio 6.x

Once the build succeeded, a runtime crash occurred:
```
TypeError: BlockContext.__init__() got an unexpected keyword argument 'allow_flagging'
```
Gradio 6.x removed/renamed this parameter from `gr.Interface`. **Fix:** removed the `allow_flagging="never"` line from the `gr.Interface(...)` call entirely (not essential functionality).

### Phase 23 — ZeroGPU / `@spaces.GPU` requirement

Next runtime error:
```
Runtime error: No @spaces.GPU function detected during startup
```
Investigated and **confirmed via live web search** that this is a genuine, currently-reported Hugging Face platform restriction (reported on HF's own forums within the last few days of this project): new free-tier accounts can only create Spaces on **ZeroGPU** hardware — Docker is paid-only, and CPU Basic **cannot be selected during Space creation**, and **downgrading an existing Space to CPU Basic requires a PRO subscription** ($9/month). This was independently confirmed by the user hitting this exact wall both when trying to downgrade an existing Space AND when trying to create a brand-new one.

**Free workaround used:** ZeroGPU Spaces only require that *some* function in the code is decorated with `@spaces.GPU` to exist at startup — it does not need to ever actually be called. Since this app does all its real computation remotely via the watsonx.ai API (no local GPU needed at all), a harmless placeholder function was added:

```python
import spaces

@spaces.GPU
def _gpu_placeholder():
    """Satisfies Hugging Face ZeroGPU's startup requirement.
    This app doesn't use local GPU — all inference runs remotely via
    the IBM watsonx.ai API — but ZeroGPU Spaces require at least one
    @spaces.GPU-decorated function to exist at startup."""
    pass
```
This function is never called anywhere; it exists purely to satisfy the startup check, and does not consume any GPU quota since it's never invoked.

Pushed:
```powershell
git add app.py
git commit -m "Add placeholder @spaces.GPU function to satisfy HF ZeroGPU startup check"
git push space main
```
**This resolved the error.**

### Phase 24 — Confirmed running

Final startup log confirmed success:
```
Running on local URL:  http://0.0.0.0:7860, with SSR (Node proxy -> Python :7861)
```
No traceback after this line — the Space is live and serving the Gradio interface.

### Phase 25 — Custom prompt template offered (⚠️ NOT CONFIRMED APPLIED)

A custom, more detailed-but-crisp RAG prompt template was drafted and offered, to replace LangChain's default internal prompt in `RetrievalQA`:

```python
from langchain.prompts import PromptTemplate

qa_prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions based only on the provided document context.

Instructions:
- Answer using only the information in the context below. If the answer isn't in the context, say so clearly instead of guessing.
- Be thorough enough to fully answer the question, but avoid unnecessary repetition or filler.
- Use short paragraphs or bullet points when listing multiple facts.
- Keep your answer focused — aim for 3-6 sentences unless the question truly requires more detail.
- Do not mention "the context" or "the document" explicitly in your answer — just answer naturally, as if you already know the information.

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"],
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_obj,
    return_source_documents=False,
    chain_type_kwargs={"prompt": qa_prompt},
)
```
**⚠️ Status unclear** — this was provided as code to add, but the conversation moved on to Git/deployment troubleshooting immediately after, and there was no confirmation this was actually integrated into `app.py`. **Verify this before assuming it's live.**

---

## 4. Confirmed Technical Facts (Critical Reference Info)

These are hard-won facts from direct trial-and-error — treat these as ground truth, not assumptions, for any future work on this project.

### 4.1 Watsonx models confirmed to work on the SN Labs `"skills-network"` project
(Confirmed via actual live HTTP 422 API error messages, not documentation)

| Model ID | Purpose |
|---|---|
| `ibm/granite-embedding-278m-multilingual` | Embeddings |
| `mistralai/mistral-small-3-1-24b-instruct-2503` | LLM — **current choice, fast & reliable** |
| `mistralai/mistral-medium-2505` | LLM — alternative |
| `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | LLM — alternative |
| `ibm/granite-4-h-small` | LLM — matches official template, alternative |

**Confirmed NOT to work** on this specific project (despite being generally valid watsonx models elsewhere): `mistralai/mixtral-8x7b-instruct-v01`, `meta-llama/llama-3-3-70b-instruct`, `ibm/slate-125m-english-rtrvr`.

**Important caveat:** this entitlement list is specific to the `project_id="skills-network"` sandbox used inside IBM's SN Labs. A real, personal watsonx.ai project (once set up — see Section 8) will likely have a **different, probably broader** set of entitled models. Don't assume this exact list applies once real credentials are in use.

### 4.2 Package/dependency landmines

- The SN Labs pinned install list did not include `chromadb` or `pypdf` — install these separately.
- `EmbeddingTypes` (from `ibm_watsonx_ai.foundation_models.utils.enums`) is for model ID references, NOT parameter keys — use `EmbedTextParamsMetaNames` (aliased as `EmbedParams`) for embedding parameters instead.
- On Hugging Face Spaces with `sdk_version: 6.20.0`, do NOT pin `gradio`, `huggingface_hub`, or `pydantic` in `requirements.txt` — let Gradio 6.x manage its own dependency chain. Pinning any of these causes a `ResolutionImpossible` build failure.
- Gradio 6.x removed the `allow_flagging` parameter from `gr.Interface()` — don't include it in code targeting Gradio 6+.

### 4.3 Hugging Face Spaces platform quirks (current as of this project, confirmed via web search)

- New free-tier HF accounts can currently **only** create Spaces on **ZeroGPU** hardware (Docker is paid-only; CPU Basic isn't selectable at creation and can't be downgraded to without PRO).
- ZeroGPU Spaces require at least one `@spaces.GPU`-decorated function to exist in the code at startup, or the Space fails to launch — even if the app never actually needs GPU compute. An unused placeholder function satisfies this.
- Binary files (images, gifs) MUST go through Git LFS. If binaries were committed *before* LFS tracking was set up, `git lfs track` alone isn't enough — you must also run `git lfs migrate import --everything` to rewrite history, then force-push.
- New Spaces auto-initialize with their own README/git history — expect an `--allow-unrelated-histories` merge on first push from an existing local repo.

### 4.4 File location conventions

**SN Labs (IBM Coursera environment):**
- Project root: `/home/project`
- Main file: `/home/project/qabot.py`
- Test script: `/home/project/test_embedding.py`
- Virtual env: `/home/project/my_env` (activate with `source my_env/bin/activate`)
- App URL pattern: `https://<username>-7860.theianext-0-labs-prod-misc-tools-us-east-0.proxy.cognitiveclass.ai/`

**Local machine (Windows) — AskMyDocs deployment repo:**
- Root: `D:\Vignesh\Projects\AskMyDocs - PDF QA_bot\`
- Contains: `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `.gitattributes`, `test_embedding.py`, `assets/demopic.png`, `assets/demogif.gif`

### 4.5 Git remotes configured on the local repo

| Remote name | URL | Notes |
|---|---|---|
| `origin` | `https://github.com/Vignesh-P-C/AskMyDocs---RAG-PDF-Q-A-bot.git` | GitHub repo. **Also has a second push URL added** (see below) — pushing to `origin` sends to BOTH GitHub and the HF Space. |
| (added push URL on origin) | `https://huggingface.co/spaces/V1gnesh/AskMyDoc` | Added via `git remote set-url --add --push origin <url>` — means `git push origin main` pushes to both places. |
| `space` | `https://huggingface.co/spaces/V1gnesh/AskMyDoc` | Separate explicit remote, also points to the HF Space. |

**⚠️ Practical implication:** because of the dual-push setup on `origin`, running `git push origin main` (or a bare `git push`, since `main` tracks `origin`) actually pushes to **both** GitHub and Hugging Face simultaneously. Running `git push space main` additionally is redundant (harmless, just extra) since `origin` already covers HF too. This isn't a problem, just worth knowing so pushes aren't assumed to be GitHub-only when they aren't.

---

## 5. Current State of All Files

### 5.1 `app.py` (best-known current state — VERIFY against actual local file before assuming exact accuracy, especially the launch() line and whether the custom prompt template from Phase 25 was added)

```python
# app.py
# AskMyDocs — RAG-based PDF Q&A bot
# Uses: LangChain, watsonx.ai LLM + Embeddings, Chroma vector store, Gradio UI

import gradio as gr
import spaces

from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

@spaces.GPU
def _gpu_placeholder():
    """Satisfies Hugging Face ZeroGPU's startup requirement.
    This app doesn't use local GPU — all inference runs remotely via
    the IBM watsonx.ai API — but ZeroGPU Spaces require at least one
    @spaces.GPU-decorated function to exist at startup."""
    pass

# ---------- 1. LLM SETUP ----------
def get_llm():
    model_id = "mistralai/mistral-small-3-1-24b-instruct-2503"

    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }

    project_id = "skills-network"  # ⚠️ ONLY WORKS INSIDE IBM SN LABS — MUST BE REPLACED FOR PUBLIC USE (see Section 7 & 8)

    llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return llm

# ---------- 2. DOCUMENT LOADER ----------
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

# ---------- 3. TEXT SPLITTER ----------
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks

# ---------- 4. EMBEDDING MODEL ----------
def watsonx_embedding():
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
        EmbedParams.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",  # ⚠️ SAME ISSUE — MUST BE REPLACED
        params=embed_params,
    )
    return watsonx_embedding

# ---------- 5. VECTOR STORE ----------
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# ---------- 6. RETRIEVER ----------
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# ---------- 7. QA CHAIN ----------
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )
    response = qa.invoke(query)
    return response["result"]

# ---------- 8. GRADIO INTERFACE ----------
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Output"),
    title="AskMyDocs — Ask Questions About Your PDF",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
)

# ---------- 9. LAUNCH ----------
if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)
```

### 5.2 `requirements.txt` (current, HF-compatible — pins that conflicted with Gradio 6.x removed)

```
ibm-watsonx-ai==1.1.2
langchain==0.2.11
langchain-community==0.2.10
langchain-ibm==0.1.11
chromadb
pypdf
```

### 5.3 `README.md` front matter (Hugging Face metadata block — verify if `python_version` and `title` were ever corrected per Phase 19's recommendations)

```yaml
---
title: AskMyDoc
emoji: 🐨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: '3.12'    # ⚠️ recommended changing to '3.11' — unconfirmed if done
app_file: app.py
pinned: false
license: mit
short_description: Stop skimming. Start asking. Turn any PDF into a convo.
---
```

### 5.4 `test_embedding.py` (standalone embedding demo script — confirmed working)

```python
# test_embedding.py
# Standalone script to demonstrate watsonx embeddings for the embedding.png screenshot

from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

def watsonx_embedding():
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
        EmbedParams.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

sentence = "How are you?"

embedding_model = watsonx_embedding()
result = embedding_model.embed_query(sentence)

print(f"Sentence: {sentence}")
print(f"First 5 embedding numbers: {result[:5]}")
```

### 5.5 `.gitignore`
```
my_env/
__pycache__/
*.pyc
.theia/
*.pdf
chroma_db/
.DS_Store
```

---

## 6. Current Deployment Status

| Component | Status |
|---|---|
| **IBM SN Labs submission** | Working end-to-end. `qabot.py` runs successfully in the lab, produces correct RAG answers. `embedding.png` and `QA_bot.png` screenshots confirmed captured correctly. Other 4 code screenshots (`pdf_loader.png`, `code_splitter.png`, `vectordb.png`, `retriever.png`) — instructions given, **not confirmed captured** in this conversation. Quiz Q10 and Q11 answered. |
| **GitHub repo** | Live at `github.com/Vignesh-P-C/AskMyDocs---RAG-PDF-Q-A-bot`. Pushed successfully, in sync with Hugging Face (via dual-push `origin` config). |
| **Hugging Face Space** | Live at `huggingface.co/spaces/V1gnesh/AskMyDoc`. Build succeeds, app starts (`Running on local URL: http://0.0.0.0:7860`), Gradio interface should be rendering in the App tab. **However: will fail at the actual Q&A step for any real visitor**, because it still uses the `project_id="skills-network"` shortcut, which only works inside IBM's SN Labs. This is the single most important remaining task — see Section 7 & 8. |
| **Custom prompt template** | Drafted, given to the user as code — **not confirmed integrated** into the live `app.py`. |
| **Custom domain / custom frontend "website"** | Not started. Explicitly discussed and deliberately deferred — the Hugging Face Space itself already counts as a live, shareable website for portfolio purposes; a custom-domain/custom-frontend version would be a distinct "v2" project, recommended only after the current version is fully working end-to-end (including real credentials). |

---

## 7. Known Open Issues / Unresolved Items

Ordered roughly by importance:

1. **🔴 CRITICAL — watsonx credentials still use the SN Labs shortcut.**
   `get_llm()` and `watsonx_embedding()` both hardcode `project_id = "skills-network"`. This ONLY authenticates inside IBM's Skills Network cloud labs. On the public Hugging Face Space, any real visitor uploading a PDF and asking a question will hit an authentication failure. **This must be fixed before the deployed Space is genuinely usable by anyone besides during active testing sessions (if those still happen to work at all outside the lab, which they likely don't).**

2. **🟡 Unconfirmed — custom prompt template integration.**
   A `PromptTemplate` was drafted (Section 3, Phase 25 / reproduced in Section 5.1's surrounding context) to make answers more detailed-but-crisp, but there's no confirmation it was actually added to `app.py`'s `retriever_qa()` function. Check the live file.

3. **🟡 Naming inconsistency.**
   GitHub repo: "AskMyDocs" (with repo folder literally named `AskMyDocs---RAG-PDF-Q-A-bot`). Hugging Face Space: "AskMyDoc" (no "s"). README body heading: "AskMyDocs". Cosmetic only, but worth standardizing on one name everywhere (Space title, README heading, GitHub repo name, tagline placement) for a cleaner portfolio presentation.

4. **🟡 `python_version: '3.12'` in the HF README.**
   The app was developed and tested exclusively on Python 3.11 (matching the SN Labs environment and the originally pinned `requirements.txt`). Recommended changing to `'3.11'` for consistency and to reduce any risk of subtle version-related bugs, though the app does appear to be running successfully as-is, so this may be a non-issue in practice.

5. **🟡 `sdk_version: 6.20.0` vs. originally-built-for `4.44.0`.**
   The app was written and tested against Gradio 4.44.0's API. It's now running on Gradio 6.20.0 after removing conflicting pins and fixing the `allow_flagging` runtime error. It's *working*, but there could be other subtle Gradio 4-vs-6 API differences not yet surfaced (untested edge cases, styling differences, etc.). Alternative not yet tried: lowering `sdk_version` back to `4.44.0` to exactly match the tested environment, which might be more stable long-term, at the cost of re-doing the dependency pin work in reverse.

6. **🟡 Uncertain: were the 4 remaining code screenshots (`pdf_loader.png`, `code_splitter.png`, `vectordb.png`, `retriever.png`) actually captured and uploaded to Coursera?**
   Instructions for exactly what to screenshot were given clearly, but no confirmation screenshots were shared back for these four specifically (unlike `embedding.png` and `QA_bot.png`, which were both confirmed). Double-check the Coursera submission is fully complete before considering the course assignment done.

7. **🟢 Minor: `launch()` arguments on Hugging Face.**
   It was recommended (but not confirmed done) to simplify `rag_application.launch(server_name="0.0.0.0", server_port=7860)` to just `rag_application.launch()` for Hugging Face Spaces, which manage ports/networking automatically. The app appears to run fine either way based on the logs, so this is low-priority — mentioned for completeness only.

---

## 8. Recommended Next Steps (Prioritized)

### Priority 1 — Fix real watsonx credentials (blocking issue for genuine public deployment)

1. Create a free **IBM Cloud account** at cloud.ibm.com if not already done.
2. Create a **watsonx.ai project** in the IBM Cloud console (Watson Studio / watsonx.ai service).
3. Generate a personal **API key** (IBM Cloud → Manage → Access (IAM) → API keys).
4. Note down the real **project ID** for the new watsonx.ai project (found in the project's settings page).
5. Update `get_llm()` and `watsonx_embedding()` in `app.py` to use proper credential objects instead of the `"skills-network"` shortcut, e.g.:
   ```python
   import os
   from ibm_watsonx_ai import Credentials

   credentials = Credentials(
       url="https://us-south.ml.cloud.ibm.com",
       api_key=os.environ.get("WATSONX_APIKEY"),
   )
   project_id = os.environ.get("WATSONX_PROJECT_ID")
   ```
   and pass `credentials=credentials` (or equivalent, depending on the exact `WatsonxLLM`/`WatsonxEmbeddings` constructor signature in the installed SDK version — verify against current `langchain-ibm` docs) instead of the bare `url=`/`project_id=` pair currently used.
6. On Hugging Face: go to the Space's **Settings → Repository secrets** → add `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` as secrets (never hardcode these in `app.py` or commit them to git).
7. **Important:** double-check whether the newly-created personal watsonx project has the same model entitlements as the SN Labs sandbox (see Section 4.1) — it may support a different, likely broader, set of models. Re-verify `model_id` choices work under the new credentials before assuming they'll behave identically.
8. Test thoroughly on the live Space with a real PDF and multiple different questions once credentials are wired up.

### Priority 2 — Confirm the Coursera submission is fully complete

1. Verify all 6 required screenshots (`pdf_loader.png`, `code_splitter.png`, `embedding.png`, `vectordb.png`, `retriever.png`, `QA_bot.png`) are actually captured and uploaded, not just planned.
2. Confirm the quiz (Question 10, Question 11, and any others in the same set) has been fully submitted.
3. Re-check the "Final Submission Guidelines and Deliverables" reading (referenced early in this project but never actually shared/reviewed in this conversation) in case it specifies anything beyond what's been covered here — e.g., whether a combined PDF of all screenshots is ever actually required (this was asked about but never confirmed either way from an authoritative source; the graded quiz interface seen so far only showed individual `.png`/`.jpeg`/`.gif`/`.webp` uploads, suggesting no PDF-combination step is actually needed, but this hasn't been 100% verified against that specific guidelines page).

### Priority 3 — Polish and consistency pass

1. Pick one final name — "AskMyDocs" or "AskMyDoc" — and make it consistent across: the Hugging Face Space title, the README heading, the GitHub repo name (renaming a GitHub repo is easy and doesn't break the remote URL if done via GitHub's rename feature, which auto-redirects the old URL).
2. Consider fixing `python_version` to `'3.11'` in the HF README for consistency with the tested environment.
3. Decide whether to keep `sdk_version: 6.20.0` (already working) or roll back to `4.44.0` (exactly matches original testing) — either is defensible; document the choice once made.

### Priority 4 — Confirm/apply the custom prompt template (optional enhancement)

Check whether the `PromptTemplate` from Phase 25 was actually added to `retriever_qa()`. If not, and if more detailed-but-crisp answers are still desired, apply it (full code is in Section 3, Phase 25).

### Priority 5 — Optional: custom website / custom frontend (deliberately deferred "v2")

Only pursue this after Priorities 1–4 are done and the current Hugging Face Space is fully working end-to-end with real credentials. This would involve:
- A custom frontend (e.g., React/Next.js) instead of Gradio's default UI
- A custom domain (e.g., `askmydocs.com`)
- Likely restructuring the backend as a proper API (e.g., FastAPI) that the custom frontend calls, rather than relying on Gradio's built-in UI

This is a genuinely separate project layered on top of a working backend — not a natural continuation of the current one. Treat it as its own effort with its own planning, not a leftover task.

---

## 9. Quick-Reference Command Cheat Sheet

### SN Labs (IBM course environment)
```bash
cd /home/project
source my_env/bin/activate
lsof -i :7860              # check if port already in use
kill -9 <PID>              # kill if needed
python3.11 qabot.py        # run the app
python3.11 test_embedding.py   # run the embedding demo script
```
App URL pattern: `https://<username>-7860.theianext-0-labs-prod-misc-tools-us-east-0.proxy.cognitiveclass.ai/`

### Local machine — pushing to both GitHub and Hugging Face
```powershell
cd "D:\Vignesh\Projects\AskMyDocs - PDF QA_bot"
git add .
git commit -m "your message here"
git push origin main     # pushes to BOTH GitHub and HF (dual-push configured on origin)
# git push space main     # redundant given the above, but harmless if run too
```

### If a future binary file (image/gif) gets rejected by Hugging Face again
```powershell
git lfs track "*.<extension>"
git add .gitattributes
git add <the-file>
git commit -m "Track new binary asset with LFS"
git push origin main
# If it's an OLD file already in history causing the rejection:
git lfs migrate import --include="*.<extension>" --everything
git push --force origin main
```

### If Hugging Face throws a new dependency conflict during build
Read the error carefully — it names the exact conflicting packages. General pattern: if a pin in `requirements.txt` conflicts with what Gradio's own dependency tree wants, remove that specific pin from `requirements.txt` and let pip resolve it. Alternative broader fix: lower `sdk_version` in the README to `4.44.0` to match the originally-tested Gradio version and avoid the whole conflict category.

---

*End of document. This log reflects everything completed in the original chat session. Cross-check Section 7's open items against the actual current state of the local files and live deployments before proceeding, since some items (prompt template integration, python_version fix, naming consistency) were recommended but never explicitly confirmed as applied.*