# backend/rag.py
# Ported RAG pipeline from app.py — same logic, no Gradio dependency.

import os
from dotenv import load_dotenv
load_dotenv()

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
            f"Set it in backend/.env for local testing, or as a Render "
            f"environment variable when deployed."
        )
    return value


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
            provider="auto",
            huggingfacehub_api_token=_require_env("HUGGINGFACEHUB_API_TOKEN"),
        )
        _llm_singleton = ChatHuggingFace(llm=base_llm)
    return _llm_singleton


def hf_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


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


def index_document(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    embedding_model = hf_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    return retriever, len(docs), len(chunks)


def answer_question(retriever, query: str) -> str:
    llm = get_llm()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query)