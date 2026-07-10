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


# Sample sentence to embed
sentence = "How are you?"

embedding_model = watsonx_embedding()
result = embedding_model.embed_query(sentence)

print(f"Sentence: {sentence}")
print(f"First 5 embedding numbers: {result[:5]}")