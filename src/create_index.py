from utils.logging.logger import get_logger
from utils.models.models import EmbeddingType
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document

logger = get_logger(__name__)

persist_directory = "../db"
logger.info("Loading documents...")
file_dir = "../docs"
models_dir = "../models/"
file = "bedrock-ug.pdf"

huggingfacehub_emb_model = "bert-base-multilingual-cased"
huggingface_emb_model = "sentence-transformers/all-mpnet-base-v2"
openai_emb_model = "text-embedding-ada-002"
collection_name = "awscollection"
embedding_model = EmbeddingType(type="huggingface")

# Define document loader
logger.info("Defining document loader...")
if file.endswith("pdf"):
    loader = PyPDFLoader(f"{file_dir}/{file}")
elif file.endswith("txt"):
    loader = TextLoader(f"{file_dir}/{file}", encoding="utf8")
documents = loader.load()

logger.info("Loading embeddings model...")
if embedding_model.type == "huggingface":
    embedding = HuggingFaceEmbeddings(
        model_name=huggingface_emb_model, cache_folder=models_dir
    )
elif embedding_model.type == "openai":
    embedding = OpenAIEmbeddings(
        model=openai_emb_model,
        show_progress_bar=True,
        max_retries=50,
        request_timeout=60 * 15,
    )

logger.info("Splitting documents...")
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, length_function=len
)
texts = text_splitter.split_documents(documents)

logger.info("Initializing vector DB...")
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name=collection_name,
)
logger.info("Indexing finished.")
