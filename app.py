# app.py
# AskMyDocs — RAG-based PDF Q&A bot
# Uses: LangChain (LCEL), Hugging Face Inference Providers (LLM, chat-based) +
# local sentence-transformers (embeddings), Chroma vector store, Gradio Blocks UI
#
# Requires this environment variable (set as a Hugging Face Space "Repository
# secret", or in a local .env file for local testing):
#   HUGGINGFACEHUB_API_TOKEN – a Fine-grained access token from
#   huggingface.co/settings/tokens with the "Inference" preset selected
#   (i.e. "Make calls to Inference Providers" permission). A plain "Read"
#   token is NOT sufficient — causes a 403 error.

import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import spaces

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it as a Hugging Face Space secret (Settings -> Variables and "
            f"secrets), or in a local .env file for testing. See .env.example."
        )
    return value


@spaces.GPU
def _gpu_placeholder():
    """Satisfies Hugging Face ZeroGPU's startup requirement.
    Text generation runs remotely via the Hugging Face Inference API, and
    embeddings run locally on CPU — this app never actually needs local GPU —
    but ZeroGPU Spaces require at least one @spaces.GPU-decorated function to
    exist at startup."""
    pass


# ---------- 1. LLM SETUP (now a cached singleton — built once, reused) ----------
_llm_singleton = None


def get_llm():
    global _llm_singleton
    if _llm_singleton is None:
        base_llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-120b",
            task="text-generation",
            max_new_tokens=768,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.03,
            provider="auto",  # let Hugging Face route to an available Inference Provider
            huggingfacehub_api_token=_require_env("HUGGINGFACEHUB_API_TOKEN"),
        )
        _llm_singleton = ChatHuggingFace(llm=base_llm)
    return _llm_singleton


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
def hf_embedding():
    # Runs locally on CPU — no API call, no Inference credits spent.
    #
    # device="cpu" is forced explicitly (not left to auto-detection).
    # On a ZeroGPU Space, torch.cuda.is_available() reports True even
    # outside an @spaces.GPU-decorated function (that's how ZeroGPU's
    # emulation works), so without this, sentence-transformers silently
    # tries to place the model on "cuda" and crashes with:
    #   "Low-level CUDA init (torch._C._cuda_init) reached... did not
    #    intercept a CUDA operation"
    # This didn't show up locally because a plain CPU-only machine never
    # reports CUDA as available in the first place.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ---------- 5. VECTOR STORE ----------
def vector_database(chunks):
    embedding_model = hf_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


# ---------- 6. QA PROMPT + CHAIN HELPERS ----------
qa_prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions based only on the provided document context.

Instructions:
- Answer using only the information in the context below. If the answer isn't in the context, say so clearly instead of guessing.
- Be thorough enough to fully answer the question, but avoid unnecessary repetition or filler.
- Use short paragraphs or bullet points when listing multiple facts.
- Keep your answer focused — aim for 3-6 sentences — UNLESS the question explicitly asks for a specific length (e.g. a word count, "in detail", "long explanation"), in which case follow that instruction instead.
- Do not mention "the context" or "the document" explicitly in your answer — just answer naturally, as if you already know the information.

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"],
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------- 7. INDEXING (runs ONCE per uploaded PDF, not once per question) ----------
def index_document(file):
    """Builds the retriever for a freshly-uploaded PDF and reports status.

    Returns (retriever_or_None, status_markdown, query_box_update, send_btn_update)
    """
    if file is None:
        return (
            None,
            "Upload a PDF to get started.",
            gr.update(interactive=False, value=""),
            gr.update(interactive=False),
        )
    try:
        docs = document_loader(file)
        chunks = text_splitter(docs)
        vectordb = vector_database(chunks)
        retriever_obj = vectordb.as_retriever()
        filename = os.path.basename(file.name)
        status = (
            f"**{filename}** indexed — {len(docs)} page(s), {len(chunks)} chunk(s). "
            f"Ask away below."
        )
        return retriever_obj, status, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        return (
            None,
            f"⚠ Couldn't index that PDF: {e}",
            gr.update(interactive=False, value=""),
            gr.update(interactive=False),
        )


# ---------- 8. ANSWERING (reuses the stored retriever + cached LLM) ----------
def respond(query, history, retriever_obj):
    history = history or []
    if not query or not query.strip():
        return history, ""

    if retriever_obj is None:
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "Please upload and index a PDF first."},
        ]
        return history, ""

    try:
        llm = get_llm()
        rag_chain = (
            {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(query)
    except Exception as e:
        answer = f"Something went wrong answering that: {e}"

    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


def reset_all():
    return (
        [],                                   # chatbot
        None,                                 # retriever_state
        "Upload a PDF to get started.",       # status_md
        gr.update(value="", interactive=False),  # query_box
        gr.update(interactive=False),         # send_btn
    )


# ---------- 9. THEME + STYLING ----------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#FBF9F4",
    body_background_fill_dark="#1F2A24",
    button_primary_background_fill="#2F4B3C",
    button_primary_background_fill_hover="#26392E",
    button_primary_text_color="#FBF9F4",
    block_border_color="#D8D2C4",
    block_title_text_color="#2F4B3C",
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&display=swap');

.askmydocs-header h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.15rem;
    color: #2F4B3C;
    margin-bottom: 0.15rem;
    letter-spacing: -0.01em;
}
.askmydocs-header p {
    color: #6B6152;
    font-size: 0.95rem;
    margin-top: 0;
}
.askmydocs-badges span {
    display: inline-block;
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    margin: 0.35rem 0.35rem 0 0;
    border: 1px solid #D8D2C4;
    border-radius: 999px;
    color: #6B6152;
    background: #FFFDF9;
}
.askmydocs-spine {
    border-left: 3px solid transparent;
    border-image: linear-gradient(180deg, #2F4B3C, #B8863B) 1;
    padding-left: 1.25rem;
}
.askmydocs-status {
    font-size: 0.85rem;
    color: #4A5A4F;
}
"""

HEADER_HTML = """
<div class="askmydocs-header">
  <h1>AskMyDocs</h1>
  <p>Upload a PDF, ask it anything. Answers come straight from your document.</p>
  <div class="askmydocs-badges">
    <span>LangChain</span><span>Hugging Face Inference</span><span>Chroma</span><span>Sentence-Transformers</span>
  </div>
</div>
"""

FOOTER_MD = (
    "<sub>Built with LangChain · Hugging Face Inference Providers · Chroma · Gradio — "
    "[source on GitHub](https://github.com/Vignesh-P-C/AskMyDocs---RAG-PDF-Q-A-bot)</sub>"
)

# ---------- 10. GRADIO INTERFACE (Blocks — replaces the old bare gr.Interface) ----------
with gr.Blocks(title="AskMyDocs") as rag_application:
    gr.HTML(HEADER_HTML)

    retriever_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Your document")
            file_upload = gr.File(
                label="Upload a PDF",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
            )
            status_md = gr.Markdown(
                "Upload a PDF to get started.", elem_classes=["askmydocs-status"]
            )
            clear_btn = gr.Button("Start over", variant="secondary", size="sm")

        with gr.Column(scale=3, elem_classes=["askmydocs-spine"]):
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(height=420, show_label=False)
            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="Ask a question about the document...",
                    show_label=False,
                    scale=5,
                    interactive=False,
                )
                send_btn = gr.Button("Ask", variant="primary", scale=1, interactive=False)

    gr.Markdown(FOOTER_MD)

    file_upload.change(
        index_document,
        inputs=file_upload,
        outputs=[retriever_state, status_md, query_box, send_btn],
    )
    send_btn.click(
        respond,
        inputs=[query_box, chatbot, retriever_state],
        outputs=[chatbot, query_box],
    )
    query_box.submit(
        respond,
        inputs=[query_box, chatbot, retriever_state],
        outputs=[chatbot, query_box],
    )
    clear_btn.click(
        reset_all,
        outputs=[chatbot, retriever_state, status_md, query_box, send_btn],
    )

# ---------- 11. LAUNCH ----------
# Note: Gradio 6 moved theme/css from the Blocks() constructor to launch().
if __name__ == "__main__":
    rag_application.launch(
        theme=theme,
        css=css,
        server_name="0.0.0.0",
        server_port=7860,
    )