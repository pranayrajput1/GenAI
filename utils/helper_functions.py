import logging
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_log():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging


def get_time(start_time, end_time):
    elapsed_time_minutes = (end_time - start_time) / 60

    logging.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logging.info(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logging.info(f"Elapsed Time: {elapsed_time_minutes:.2f}")


def get_memory_usage():
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 3)  # Convert bytes to gigabytes
    used_memory = memory.used / (1024 ** 3)
    available_memory = memory.available / (1024 ** 3)
    percentage_used = memory.percent

    logging.info(f"Total Memory: {total_memory:.2f} GB")
    logging.info(f"Used Memory: {used_memory:.2f} GB")
    logging.info(f"Available Memory: {available_memory:.2f} GB")
    logging.info(f"Percentage Used: {percentage_used}%")


logging.info("Task: Defining special tokens to be to be used in model training")
"""To be added as special tokens"""
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

logging.info("Task: Defining load tokenizer function")
"""This is the function to load tokenizer"""


def load_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    logging.debug("Memory usage in loading model tokenizer")
    get_memory_usage()
    return tokenizer


logging.info("Task: Defining load model function")
"""This is the function to load model"""


def load_model(pretrained_model_name_or_path, gradient_checkpointing):
    default_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True
    )
    logging.debug("Memory usage in loading model")
    get_memory_usage()
    return default_model


logging.info("Task: Defining function to get model & tokenizer")
"""This is the function to call for loading both tokenizer and model"""


def get_model_tokenizer(
        pretrained_model_name_or_path, gradient_checkpointing):
    pretrained_tokenizer = load_tokenizer(pretrained_model_name_or_path)
    pretrained_model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing
    )
    pretrained_model.resize_token_embeddings(len(pretrained_tokenizer))
    return pretrained_model, pretrained_tokenizer
