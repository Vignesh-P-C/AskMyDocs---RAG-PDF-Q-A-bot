# qabot.py
# Final Project: Build a QA Bot to Read Your Document
# Uses: LangChain, watsonx.ai LLM + Embeddings, Chroma vector store, Gradio UI

import gradio as gr

from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

import spaces

@spaces.GPU
def _gpu_placeholder():
    """Satisfies Hugging Face ZeroGPU's startup requirement.
    This app doesn't use local GPU — all inference runs remotely via
    the IBM watsonx.ai API — but ZeroGPU Spaces require at least one
    @spaces.GPU-decorated function to exist at startup."""
    pass

# ---------- 1. LLM SETUP ----------
def get_llm():
    model_id = "mistralai/mistral-small-3-1-24b-instruct-2503"  # reverted: faster and confirmed working for this project

    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }

    project_id = "skills-network"  # provided automatically inside SN Labs

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
        project_id="skills-network",
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