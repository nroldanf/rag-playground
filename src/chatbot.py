import os
from utils.logging.logger import get_logger
from utils.models.models import EmbeddingType, LLMType
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains import VectorDBQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
)
import torch
import mlflow

os.environ["TRANSFORMERS_CACHE"] = "../models"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import pipeline

logger = get_logger(__name__)

persist_directory = "../db"
models_dir = "../models"
huggingface_emb_model = "sentence-transformers/all-mpnet-base-v2"
openai_emb_model = "text-embedding-ada-002"
collection_name = "awscollection"
# https://huggingface.co/databricks/dolly-v2-7b
llm_model = "databricks/dolly-v2-3b"

tracking_uri = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment(
    experiment_name=f"{llm_model}-{huggingface_emb_model}"
)

embedding_type = EmbeddingType(type="huggingface")
llm_type = LLMType(type="huggingface")

logger.info("Loading embedding model...")
if embedding_type.type == "huggingface":
    embedding = HuggingFaceEmbeddings(
        model_name=huggingface_emb_model,
        cache_folder=f"{models_dir}/{huggingface_emb_model}",
    )
elif embedding_type.type == "openai":
    embedding = OpenAIEmbeddings(
        model=openai_emb_model,
        show_progress_bar=True,
        max_retries=50,
        request_timeout=60 * 15,
    )

logger.info("Initializing vectordb...")
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name=collection_name,
)

logger.info("Loading LLM model...")

if llm_type.type == "huggingface":
    generate_text = pipeline(
        model=llm_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        return_full_text=True,
    )
    llm = HuggingFacePipeline(pipeline=generate_text)
elif llm_type.type == "openai":
    llm = OpenAI(model_name="text-davinci-003")

query = "What are the pricing models bedrock has?"

logger.info("Searching for relevant docs...")
docs = vectordb.similarity_search(query=query, k=2, filter={})
logger.info(f"docs: {docs}")

# Prompt template with context
prompt_template = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}",
)

logger.info("Initializing LLM chain...")
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

with mlflow.start_run(experiment_id=experiment.experiment_id):
    context = "".join([doc.page_content for doc in docs])
    result = llm_chain.predict(instruction=query, context=context)
    inputs = [{"question": query, "context": context}]

    # mlflow.langchain.log_model(
    #     lc_model=llm_chain,
    #     artifact_path=f"{llm_model}-{huggingface_emb_model}",
    #     registered_model_name=f"{llm_model}-{huggingface_emb_model}"
    # )

    mlflow.llm.log_predictions(
        inputs=inputs, outputs=[result], prompts=[prompt_template.template]
    )
    if embedding_type.type == "huggingface":
        mlflow.log_param("embedding", huggingface_emb_model)
    elif embedding_type.type == "openai":
        mlflow.log_param("embedding", openai_emb_model)

    mlflow.log_param("llm", llm_model)
    mlflow.log_param("context collection", collection_name)

    logger.info(f"Prediction from LLM: {result}")
