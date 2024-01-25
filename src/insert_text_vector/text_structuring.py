import glob
import time

from src.model.model import get_response
from src.templates.structuring_template import string_template, task, decorator
from src.utils.constants import structured_text_dir
from src.utils.helpers import setup_logger, get_time, read_pdf


def process_resumes_structuring(resume_directory, input_model, input_tokenizer, model_reload_state):
    logger = setup_logger()
    try:
        logger.info("Performing resume structuring")

        path_list = glob.glob(f'{resume_directory}/*.pdf')
        logger.info(path_list)

        for resume in path_list:
            extracted_resume_text = read_pdf(resume)
            filename = resume.split('/')[-1].split('.')[0]
            logger.info("Processing Resume: ", filename)
            input_template = string_template.format(TASK=task,
                                                    RESUME=extracted_resume_text,
                                                    DECORATOR=decorator)
            start_time = time.time()
            output_mistral = get_response(user_query=input_template,
                                          model=input_model,
                                          tokenizer=input_tokenizer,
                                          reload_state=model_reload_state)
            # logger.info(output_mistral)
            string_value = output_mistral['predictions'].split('[/INST]')[1]
            # logger.info(output_mistral['predictions'].split('[/INST]')[1])
            end_time = time.time()
            get_time(start_time, end_time)
            file1 = open(f"{structured_text_dir}/{filename}_skills_list_up.txt", "w")
            file1.write(string_value)
            file1.close()
            logger.info("File Written Successfully \n")
        return "Resume Structured Successfully", 200
    except Exception as e:
        return f"Some error occurred in resume structuring, error: {str(e)}", 505
