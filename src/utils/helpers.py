import time
import logging
from google.cloud import storage
import PyPDF2
import requests
import json
from src.retriever.retriever import retriever
from src.utils.constants import local_instance_endpoint_url


def setup_logger():
    """
    Initializing logger basic configuration
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging


logger = setup_logger()


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

            Provide a list of technical skills for each candidate. Use format: ["skill_1", "skill_2", ...]"""

            output_mistral = local_inference_point(input_prompt=model_input_prompt)
            return output_mistral
    except Exception as e:
        return f"Some error occurred in ranking resumes, error: {str(e)}"

