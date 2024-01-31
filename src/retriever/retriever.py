from utils.helpers import setup_logger
from utils.constants import embeddings_model, preprocessing_dir
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from insert_text_vector.text_structuring import local_inference_point

from utils.constants import persistence_directory

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                  persist_directory=persistence_directory
                                  ))
embedding_functions = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embeddings_model)

collection2 = client.get_or_create_collection(name="test_one", embedding_function=embedding_functions)

logger = setup_logger()


def retriever(user_prompt):
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


def get_ranking_resumes(job_title,
                        desired_skills):
    vdb_prompt = f"Find resumes containing these mentioned skills {desired_skills}"

    logger.info(f"Fetching resume vectors data from chroma db")
    resumes = retriever(user_prompt=vdb_prompt)

    if len(resumes) > 0:
        logger.info(f"Retrieved Resumes:{resumes}")

        model_input_prompt = f"""IT HR Recruiter Task: Rank resumes for '{job_title}' based on:
        1. Job Title: '{job_title}'
        2. Skills: {desired_skills}

        Rank resumes by assessing candidates' proficiency in the specified skills. 
        Resumes: {resumes}

        Provide a list of technical skills for each candidate. Use format: ["skill_1", "skill_2", ...]"""

        output_mistral = local_inference_point(input_prompt=model_input_prompt)
        return output_mistral
