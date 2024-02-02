import time
from google.cloud import storage
import PyPDF2
import requests
import json
from utils.constants import local_instance_endpoint_url
from chromadb.utils import embedding_functions
from utils.constants import embeddings_model
import chromadb
from chromadb.config import Settings
from utils.constants import persistence_directory
import logging


def setup_logger():
    """
    Initializing logger basic configuration
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging


logger = setup_logger()


def get_embeddings_function(model: str):
    """Function to return embeddings function based on provide model name as input"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)


def get_time(start_time_input, end_time_input):
    elapsed_time_minutes = (start_time_input - end_time_input) / 60

    logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_input))}")
    logger.info(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_input))}")
    logger.info(f"Elapsed Time: {elapsed_time_minutes:.2f}")


def create_directory_if_not(directory_path):
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory '{directory_path}' created.")
    else:
        logger.info(f"Directory '{directory_path}' already exists.")


def download_files_from_bucket(bucket_name, destination_folder):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()

        files_downloaded = 0

        for blob in blobs:
            if blob.name.lower().endswith(('.pdf', '.doc', '.docx')):
                destination_path = f"{destination_folder}/{blob.name}"
                blob.download_to_filename(destination_path)
                files_downloaded += 1
                logger.info(f"Downloaded {blob.name} to {destination_path}")

        if files_downloaded > 0:
            return True, 200
        else:
            return f"No files found in the bucket: {bucket_name}", 404

    except Exception as e:
        return logger.error(f"Some error occurred in downloading resume from bucket, error: {str(e)}")


def read_pdf(filepath):
    pdf_reader = PyPDF2.PdfReader(filepath)
    num_page = len(pdf_reader.pages)
    logger.info(f'Total Number of pages in the PDF:{num_page}')
    texts = []

    for page_no in range(num_page):
        page_object = pdf_reader.pages[page_no]
        texts.append(page_object.extract_text())

    text_example = '\n'.join(texts)
    return text_example


def local_inference_point(input_prompt):
    logger.info("I am in predict endpoint.")
    data = {"input": input_prompt, "model_state": False}
    response = requests.post(url=local_instance_endpoint_url, json=data)
    logger.info(f"Mistral Predict Endpoint Status Code:{response.status_code}")
    return json.loads(response.text)


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
    logging.info(indexes)
    # threshold_docs = ["\n Resume-{}:\n".format(i+1)+results['documents'][0][index] for i, index in enumerate(indexes)]
    threshold_docs = ["\n Resume ID: {}:\n".format(results["ids"][0][index]) + results['documents'][0][index] for index
                      in
                      indexes]
    retrieved_documents = '\n'.join(threshold_docs)
    # logger.info(retrieved_documents)
    return retrieved_documents


def get_ranking_resumes(job_title,
                        desired_skills):
    try:
        vdb_prompt = f"Find resumes containing these mentioned skills {desired_skills}"

        logger.info(f"Fetching resume vectors data from chroma db")
        resumes = retriever(user_prompt=vdb_prompt)
        logger.info(f"Retrieved Resumes: {resumes}")

        if len(resumes) > 0:
            logger.info(f"Retrieved Resumes:{resumes}")

            model_input_prompt = f"""IT HR Recruiter Task: Rank resumes for '{job_title}' based on:
            1. Job Title: '{job_title}'
            2. Skills: {desired_skills}

            Rank resumes by assessing candidates' proficiency in the specified skills. 
            Resumes: {resumes}
            
            Also State the reasons of ranking those resumes corresponding to them.

            Provide a list of technical skills for each candidate. Use format: ["skill_1", "skill_2", ...]"""

            output_mistral = local_inference_point(input_prompt=model_input_prompt)
            return output_mistral
    except Exception as e:
        return f"Some error occurred in ranking resumes, error: {str(e)}"
