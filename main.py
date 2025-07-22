from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Convert datasets into a list of strings
def load_datasets(dataset_path):
    with open(dataset_path) as f:
        text = [l.strip() for l in f if l.strip()]
        return text 
# Set up the device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForCausalLM.from_pretrained('bert-base-uncased',output_hidden_states=True ,is_decoder=True).to(device)
model.eval()
model.to(device)
# Function to tokenize and batch the text
def tokenize_and_batch(text , batch_size: int = 16):
    for i in range(0 , len(text) , batch_size):
        batch_texts = text[i:i+batch_size]
        batch_tokens = tokenizer(batch_texts , padding=True , truncation=True , return_tensors='pt').to(device)
        yield batch_tokens.to(device)
# Function to compute means of hidden states
def compute_means(text:list[str] , batch_size: int = 16):
    sums , total_tokens = {} , 0
    for batch_token in tokenize_and_batch(text , batch_size):
        with torch.no_grad():
            outputs = model(**batch_token)
            hidden_states = outputs.hidden_states
            
            mask = batch_token['attention_mask']
            n_tok = mask.sum().item()
            total_tokens += n_tok

            for layer_idx, layer in enumerate(hidden_states):
                summed = (layer*mask[...,None]).sum(dim=(0,1)).cpu()
                sums[layer_idx] = sums.get(layer_idx, 0) + summed
    return [sums[i]/total_tokens for i in sums]
# Function to calculate the steering vector``
def calculate_steering_vector(text_a, text_b, batch_size=16):
    means_a = compute_means(text_a, batch_size)
    means_b = compute_means(text_b, batch_size)
    
    steering_vector = [a - b for a, b in zip(means_a, means_b)]
    return steering_vector

                    



    

