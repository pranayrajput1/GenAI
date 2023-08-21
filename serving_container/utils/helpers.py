import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from transformers import PreTrainedTokenizer
from google.cloud import storage

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
        use_cache=False if gradient_checkpointing else True,
        offload_folder="offload",
    )
    return model


def get_model_tokenizer(
        pretrained_model_name_or_path,
        gradient_checkpointing
):
    pretrained_tokenizer = load_tokenizer(pretrained_model_name_or_path)
    pretrained_model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing
    )
    pretrained_model.resize_token_embeddings(len(pretrained_tokenizer))
    get_memory_usage()
    return pretrained_model, pretrained_tokenizer


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        ValueError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    logging.error(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def preprocess(tokenizer, instruction_text):
    prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(
        instruction=instruction_text
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", )
    inputs["prompt_text"] = prompt_text
    inputs["instruction_text"] = instruction_text
    return inputs


def forward(model, tokenizer, model_inputs, max_length=200):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]

    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length
    )

    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(
        in_b, out_b // in_b, *generated_sequence.shape[1:]
    )
    instruction_text = model_inputs.get("instruction_text", None)
    get_memory_usage()
    return {
        "generated_sequence": generated_sequence,
        "input_ids": input_ids, "instruction_text": instruction_text
    }


def postprocess(tokenizer, model_outputs, return_full_text=False):
    response_key_token_id = get_special_token_id(tokenizer, RESPONSE_KEY_NL)
    end_key_token_id = get_special_token_id(tokenizer, END_KEY)
    generated_sequence = model_outputs["generated_sequence"][0]
    instruction_text = model_outputs["instruction_text"]
    generated_sequence = generated_sequence.numpy().tolist()
    records = []

    print(response_key_token_id, end_key_token_id)

    for sequence in generated_sequence:
        decoded = None

        try:
            response_pos = sequence.index(response_key_token_id)
        except ValueError:
            logging.error(
                f"Could not find response key {response_key_token_id} in: {sequence}"
            )
            response_pos = None

        if response_pos:
            try:
                end_pos = sequence.index(end_key_token_id)
            except ValueError:
                # logger.warning(
                #     f"Could not find end key, the output is truncated!"
                # )
                print("Could not find end key, the output is truncated!")
                end_pos = None
            decoded = tokenizer.decode(
                sequence[response_pos + 1: end_pos]).strip()

        # If True,append the decoded text to the original instruction.
        if return_full_text:
            decoded = f"{instruction_text}\n{decoded}"
        rec = {"generated_text": decoded}
        records.append(rec)

    get_memory_usage()
    return records


def download_model_files_from_bucket(bucket_name, destination_folder):
    # Initialize a GCS client
    client = storage.Client()

    # Get the desired bucket
    bucket = client.get_bucket(bucket_name)

    # List all files in the bucket
    blobs = bucket.list_blobs()

    for blob in blobs:
        # Construct the local file path
        local_path = f"{destination_folder}/{blob.name}"

        # Download the file
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded: {blob.name} to {local_path}")
