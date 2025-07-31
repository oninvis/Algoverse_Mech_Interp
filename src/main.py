from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Iterator

# Function to tokenize and batch the text
def tokenize_and_batch(tokenizer: AutoTokenizer, text: list[str], batch_size: int = 16, device='cpu') -> Iterator[list[str]]:
    # For all of our texts,
    for i in range(0, len(text), batch_size):
        # Split each piece of text into its respective batch (e.g. 0-31, 32-63 if batch-size was 32, etc.)
        batch_texts = text[i:i + batch_size]
        batch_tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        yield batch_tokens


# Function to compute means of hidden states
def compute_means(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: list[str], batch_size: int = 16, device='cpu') -> list[torch.Tensor]:
    sums: list[torch.Tensor] = []
    total_tokens: int = 0

    for batch_token in tokenize_and_batch(tokenizer, text, batch_size, device):  # Process text in batches
        with torch.no_grad():  # Disable gradient computation because we're not training (saves memory and compute)

            outputs = model(**batch_token)
            hidden_states = outputs.hidden_states

            # Get attention mask to identify real tokens vs padding
            mask = batch_token['attention_mask']
            # Count actual (non-padded) tokens in this batch
            n_tok = mask.sum().item()
            total_tokens += n_tok

            # For each layer in the model
            for layer in hidden_states:
                # Apply mask to exclude padding tokens and sum across batch and sequence dimensions
                # mask[...,None] expands mask to have the same # of dimensions as the layer, by adding a None hidden state dimension
                # mask is filled with 1's/0's, each representing if a token is real (1) or padding (0)
                # Multiplying layer by mask[***,None] multiplies the 1's by real tokens and 0's by padding, zeroing out the effect of padding tokens.
                # sum(dim = (0,1)) sums accross both dimension 0 and 1 (layers have a batch dimension (dim 0), sequence length dimension (dim 1), and a hidden size dimension (dim 2, not summed up)
                # And the resulting sum from that is a vector of size [hidden_size] because we don't sum along that dimension
                # .cpu() moves the operation to the cpu
                sums.append((layer * mask[..., None]).sum(dim=(0, 1)).cpu())
                # Condensed this from a dict to an array because a dict that points consistently increasing indexes to values is just a list with extra steps and is unnecessary memory usage
    # Divide by the total tokens to get a mean direction
    # TODO: QUESTION: Why are we dividing by total tokens and not the tokens of each specific batch? Verify that this is the right thing to do
    return [s / total_tokens for s in sums]


# Function to calculate the steering vector``
def calculate_steering_vector(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text_a, text_b, batch_size=16, device='cpu') -> list[torch.Tensor]:
    means_a = compute_means(model, tokenizer, text_a, batch_size, device)
    means_b = compute_means(model, tokenizer, text_b, batch_size, device)

    steering_vector = [a - b for a, b in zip(means_a, means_b)]
    return steering_vector
