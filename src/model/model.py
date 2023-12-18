from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
import torch
from utils.constants import model_id

torch.cuda.empty_cache()


def get_model_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    auth_token = "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"
    device_map = "auto"

    print("Loading model")
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        token=auth_token
    )

    print("Loading tokenizer")
    loaded_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=auth_token
    )
    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    return loaded_model, loaded_tokenizer


model, tokenizer = get_model_tokenizer(model_id)


def generate_text(inputs):
    model_inputs = tokenizer.apply_chat_template(inputs, return_tensors="pt")
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded


def get_response(user_query):
    messages = [
        {"role": "user", "content": f'{user_query}'}
    ]
    model_response = generate_text(messages)
    return model_response
