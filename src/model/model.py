from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
import torch
from utils.constants import model_id, auth_token

torch.cuda.empty_cache()


def get_model_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
	bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = "auto"

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        token=auth_token,
        #torch_dtype=torch.bfloat16
    )

    loaded_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=auth_token
    )
    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    return loaded_model, loaded_tokenizer


def reload_model(model_name: str):
    print("Reloading model and tokenizer again")
    try:
        reloaded_model, reloaded_tokenizer = get_model_tokenizer(model_name)
        return reloaded_model, reloaded_tokenizer
    except Exception as e:
        print(f"Error reloading model: {e}")
        return None, None


def generate_text(inputs, model, tokenizer, reload_model_state: bool):
    if reload_model_state:
        reloaded_model, reloaded_tokenizer = reload_model(model_id)
        if reloaded_model and reloaded_tokenizer:
            model, tokenizer = reloaded_model, reloaded_tokenizer

    model_inputs = tokenizer.apply_chat_template(inputs, return_tensors="pt")
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded


def get_response(user_query, model, tokenizer, reload_state=None):
    user_input = [
        {"role": "user", "content": f'{user_query}'}
    ]
    model_response = generate_text(user_input, model, tokenizer, reload_model_state=reload_state)
    return model_response
