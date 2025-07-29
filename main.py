from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Convert datasets into a list of strings
def load_datasets(dataset_path):
    with open(dataset_path) as f:
        text = [l.strip() for l in f if l.strip()]
        return text 

# Set up the device and model
#Use cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Using bert-base-uncased for now: https://huggingface.co/google-bert/bert-base-uncased
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForCausalLM.from_pretrained('bert-base-uncased',output_hidden_states=True,is_decoder=True).to(device)

#Sets up model for use instead of training
model.eval()
model.to(device) #Migrate model to GPU/CPU as needed

# Function to tokenize and batch the text
def tokenize_and_batch(text , batch_size: int = 16):
    #For all of our texts,
    for i in range(0 , len(text) , batch_size):
        #Split each piece of text into its respective batch (e.g. 1-32, 33-64 if batch-size was 32, etc.)
        batch_texts = text[i:i+batch_size]
        batch_tokens = tokenizer(batch_texts , padding=True , truncation=True , return_tensors='pt').to(device)
        #using yield instead of return turns this function into an iterator
        yield batch_tokens

# Function to compute means of hidden states
def compute_means(text:list[str] , batch_size: int = 16):
    # Initialize dictionary to store layer sums and counter for total tokens (so we can divide for avg later)
    #Note: I (farhan) switched this from a dictionary to an array, because the dictionary was being used like an array
    #What I mean by this is that the function used keys from 0...n-1 in order as keys, and then used list comprehension afterwards to return a list anyways
    # tldr I think gpt got a little confused and accidentally made a dictionary so it's an array now
    sums , total_tokens = [], 0
    
    for batch_token in tokenize_and_batch(text , batch_size): # Process text in batches
        with torch.no_grad(): # Disable gradient computation because we're not training (saves memory and compute)
            # Get model outputs including hidden states from all layers
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
                # mask[...,None] expands mask to have the same # of dimensions as the layer, by adding an None hidden state dimension
                # mask is filled with 1's/0's, each representing if a token is real (1) or padding (0)
                # Multiplying layer by mask[***,None] multiplies the 1's by real tokens and 0's by padding, zeroing out the effect of padding tokens.
                # sum(dim = (0,1)) sums accross both dimension 0 and 1 (layers have a batch dimension (dim 0), sequence length dimension (dim 1), and a hidden size dimension (dim 2, not summed up)
                #And the resulting sum from that is a vector of size [hidden_size] because we don't sum along that dimension
                # .cpu() moves the operation to the cpu
                sums.append((layer*mask[...,None]).sum(dim=(0,1)).cpu())
                #Condensed this from a dict to an array because a dict that points consistently increasing indexes to values is just a list with extra steps and is unnecessary memory usage
    #Divide by the total tokens to get a mean direction
    #TODO: QUESTION: Why are we dividing by total tokens and not the tokens of each specific batch? Verify that this is the right thing to do
    return [sums[i]/total_tokens for i in sums]


# Function to calculate the steering vector``
def calculate_steering_vector(text_a, text_b, batch_size=16):
    means_a = compute_means(text_a, batch_size)
    means_b = compute_means(text_b, batch_size)
    
    steering_vector = [a - b for a, b in zip(means_a, means_b)]
    return steering_vector

                    



    

