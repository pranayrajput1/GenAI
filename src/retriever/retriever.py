from utils.helpers import setup_logger
from utils.constants import embeddings_model
import chromadb
from chromadb.config import Settings
from utils.constants import persistence_directory
from insert_text_vector.chroma_db_impl import get_embeddings_function

logger = setup_logger()


def retriever(user_prompt):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                      persist_directory=f"./{persistence_directory}"
                                      ))
    embedding_function = get_embeddings_function(embeddings_model)

    collection2 = client.get_or_create_collection(name="test_one", embedding_function=embedding_function)
    results = collection2.query(
        query_texts=[user_prompt],
        n_results=5
    )
    indexes = [index for index in range(len(results['distances'][0])) if results['distances'][0][index] >= 0.10]
    logger.info(indexes)
    # threshold_docs = ["\n Resume-{}:\n".format(i+1)+results['documents'][0][index] for i, index in enumerate(indexes)]
    threshold_docs = ["\n Resume ID: {}:\n".format(results["ids"][0][index]) + results['documents'][0][index] for index
                      in
                      indexes]
    retrieved_documents = '\n'.join(threshold_docs)
    # logger.info(retrieved_documents)
    return retrieved_documents
