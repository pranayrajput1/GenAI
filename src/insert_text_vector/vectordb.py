import os
from langchain.document_loaders import JSONLoader, TextLoader
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader

from utils.constants import structured_text_dir, sentence_transformer_model
from utils.helpers import setup_logger

logger = setup_logger()

loader = DirectoryLoader(structured_text_dir,
                         glob="*.txt",
                         loader_cls=TextLoader)
loader_single = TextLoader("/src/data/structured_text/sample.txt")
documents = loader.load()
db = FAISS.from_documents(documents,
                          HuggingFaceEmbeddings(model_name=sentence_transformer_model))

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

query = "Find skills"
# docs = db.similarity_search(query)
docs = db.similarity_search_with_score(query)
logger.info(len(docs))
# logger.info(docs[0].page_content)
