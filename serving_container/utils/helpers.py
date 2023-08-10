import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)


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


INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"


def load_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
    )
    get_memory_usage()
    return tokenizer


def load_model(pretrained_model_name_or_path, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
        pretrained_model_name_or_path, gradient_checkpointing):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))
    get_memory_usage()
    return model, tokenizer
