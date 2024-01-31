import glob
import time
from templates.structuring_template import string_template, task, decorator
from utils.constants import structured_text_dir
from utils.helpers import setup_logger, get_time, read_pdf

from utils.helpers import local_inference_point


def process_resumes_structuring(resume_directory):
    logger = setup_logger()
    try:
        logger.info("Performing resume structuring")

        path_list = glob.glob(f'{resume_directory}/*.pdf')
        logger.info(path_list)

        for resume in path_list:
            extracted_resume_text = read_pdf(resume)
            filename = resume.split('/')[-1].split('.')[0]
            logger.info(f"Processing Resume: {filename}")
            input_template = string_template.format(TASK=task,
                                                    RESUME=extracted_resume_text,
                                                    DECORATOR=decorator)
            start_time = time.time()
            output_mistral = local_inference_point(input_prompt=input_template)
            logger.info(output_mistral)

            string_value = output_mistral['predictions'].split('[/INST]')[1]
            logger.info(output_mistral['predictions'].split('[/INST]')[1])
            end_time = time.time()
            get_time(start_time, end_time)

            file1 = open(f"{structured_text_dir}/{filename}_skills_list_up.txt", "w")
            file1.write(string_value)
            file1.close()
            logger.info("File Written Successfully \n")
        return "Resume Structured Successfully", 200

    except Exception as e:
        return f"Some error occurred in resume structuring, error: {str(e)}", 505
