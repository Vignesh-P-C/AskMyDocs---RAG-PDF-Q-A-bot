# app.py
# AskMyDocs — RAG-based PDF Q&A bot
# Uses: LangChain, Hugging Face Inference API (LLM) + local sentence-transformers
# (embeddings), Chroma vector store, Gradio UI
#
# Requires this environment variable (set as a Hugging Face Space "Repository
# secret", or exported in your shell for local testing):
#   HUGGINGFACEHUB_API_TOKEN – an access token from huggingface.co/settings/tokens
# (See .env.example for a reference template.)

import os
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import spaces

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it as a Hugging Face Space secret (Settings -> Variables and "
            f"secrets), or export it locally for testing. See .env.example."
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

# ---------- 1. LLM SETUP ----------
def get_llm():
    base_llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
        repetition_penalty=1.03,
        provider="auto",
        huggingfacehub_api_token=_require_env("HUGGINGFACEHUB_API_TOKEN"),
    )
    return ChatHuggingFace(llm=base_llm)

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
    # Runs locally on the Space's CPU — no API call, no Inference credits spent.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- 5. VECTOR STORE ----------
def vector_database(chunks):
    embedding_model = hf_embedding()
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    rag_chain = (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)
    return response

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