from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
import torch

DEVICE = torch.device('mps')

def get_model(model_name):
    model = HookedTransformer.from_pretrained_no_processing(model_name, device = DEVICE, dtype=torch.float16, default_padding_side='left', output_hidden_states=True)
    model.eval()
    model.to(DEVICE)
    return model

model = get_model("Qwen/Qwen1.5-1.8B-Chat")

def tokenize_prompt(model: HookedTransformer, prompt_str: str) -> str:
    
    prompt_message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_str}
    ]

    prompt_chat_tokenized = model.tokenizer.apply_chat_template(
        prompt_message, tokenize=True, add_generation_prompt=True)
    prompt_chat_str = model.tokenizer.apply_chat_template(
        prompt_message, tokenize=False, add_generation_prompt=True)
        
    return prompt_chat_tokenized, prompt_chat_str

def generate_output(model: HookedTransformer, prompt_chat_str: str, max_new_tokens: int) -> tuple[str, dict, int]:
    """Generate output string, cache, and number of tokens generated."""
    output_str = prompt_chat_str
    for i in tqdm(range(max_new_tokens)):
        # Get the logits and cache for the current prompt
        logits, cache = model.run_with_cache(output_str)

        # Get the predicted next token (using argmax for temperature 0)
        next_token = logits[0, -1].argmax()

        # Convert the next token to a string
        next_token_str = model.to_string(next_token)

        # Append the new token to the prompt for the next iteration
        output_str += next_token_str
        
        if next_token.item() == model.tokenizer.eos_token_id:
            break
    
    return output_str

def normal_generation(model, prompt, max_tokens):
    _, pt = tokenize_prompt(model, prompt)
    base_gen = generate_output(model, pt, max_tokens)
    return base_gen

print(normal_generation(model, "Hello!", 32))