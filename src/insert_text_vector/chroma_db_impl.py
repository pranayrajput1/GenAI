import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from chromadb.utils import embedding_functions
from src.utils.constants import structured_text_dir, embeddings_model, persistence_directory
from src.utils.helpers import setup_logger

logger = setup_logger()


def get_embeddings_function(model: str):
    """Function to return embeddings function based on provide model name as input"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)


def resume_vec_insert(persist_directory, structured_resume_dir):
    """Function to insert structured resumes data into vector db"""
    try:
        logger.info("Initialising vectordb connection")
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                          persist_directory=f"./{persist_directory}"
                                          ))

        logger.info("Loading files from directory: connection")
        loader = DirectoryLoader(structured_resume_dir,
                                 glob="*_skills_list_up.txt",
                                 loader_cls=TextLoader)
        documents = loader.load()

        input_data = [[doc.page_content for doc in documents],
                      [doc.metadata for doc in documents],
                      [doc.metadata['source'].split('/')[-1].split('.')[0] for doc in documents]]

        logger.info(f"Loading embedding function of model: {embeddings_model}")
        embeddings_function = get_embeddings_function(embeddings_model)

        collection2 = client.get_or_create_collection(name="test_one", embedding_function=embeddings_function)
        documents_embeddings = embeddings_function(input_data[0])
        logger.info(documents_embeddings)
        collection2.add(embeddings=documents_embeddings,
                        documents=input_data[0],
                        metadatas=input_data[1],
                        ids=input_data[2])
        logger.info(f"Inserted structured resumes data into vector db successfully")
        client.persist()
        return "Updated Vector Db Successfully", 200
    except Exception as e:
        return logger.error(f"Some error occurred in entering structured resumes into vector db, error: {str(e)}")

# if __name__ == "__main__":
#     resume_vec_insert(persistence_directory, structured_text_dir)
