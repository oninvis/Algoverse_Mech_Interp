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


#Farhan - Added a function to prompt the LLM
def prompt_model(batched_tokens) -> str: #I have no idea what the datatype of the input is for this tbh

    #Model generates the response based on the inputted tokens
    generated_ids = model.generate(
      batched_tokens.input_ids,
      max_new_tokens=512,
      generation_config=GenerationConfig(num_beams = 1, do_sample = False)
    ) #Do_Sample = False --> Greedy Decoding

    #Transform to decodable format
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(batched_tokens.input_ids, generated_ids)
    ]

    #Have the tokenizer decode it
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

#Checks the neutrality of a response by checking how many neutral phrases it responds with
def check_neutrality(response: str) -> int:
    #Neutral phrase list borrowed from BBQ source code
    unknowns = ["unknown", "cannot be determined", "can't be determined", "not answerable",
             "not known", "not enough info", "not enough information", "cannot answer",
             "can't answer", "undetermined"]
    unknown_count = 0
    for unk in unknowns:
        unknown_count += response.count(unk)
    return unknown_count

# Function to compute means of hidden states
def compute_means(text:list[str] , batch_size: int = 16) -> tuple(list, list):
    # Initialize dictionary to store layer sums and counter for total tokens (so we can divide for avg later)
    #Note: I (farhan) switched this from a dictionary to an array, because the dictionary was being used like an array
    #What I mean by this is that the function used keys from 0...n-1 in order as keys, and then used list comprehension afterwards to return a list anyways
    # tldr I think gpt got a little confused and accidentally made a dictionary so it's an array now
    neutral_sums, biased_sums , neutral_tokens, biased_tokens = [], [], 0, 0
    
    for batch_tokens in tokenize_and_batch(text , batch_size): # Process text in batches
        with torch.no_grad(): # Disable gradient computation because we're not training (saves memory and compute)
            # Get model outputs including hidden states from all layers
            outputs = model(**batch_tokens)
            neutrality = check_neutrality(prompt_model(batch_tokens))
            hidden_states = outputs.hidden_states
            
            # Get attention mask to identify real tokens vs padding
            mask = batch_tokens['attention_mask']
            # Count actual (non-padded) tokens in this batch
            n_tok = mask.sum().item()

            if neutrality == 1: #Answer too wishy-washy to be used as a datapoint -> ignore.
                continue
            elif neutrality > 1:
                neutral_tokens += n_tok
            else:
                biased_tokens += n_tok

            # For each layer in the model
            for layer in hidden_states:
                # Apply mask to exclude padding tokens and sum across batch and sequence dimensions
                # mask[...,None] expands mask to have the same # of dimensions as the layer, by adding an None hidden state dimension
                # mask is filled with 1's/0's, each representing if a token is real (1) or padding (0)
                # Multiplying layer by mask[***,None] multiplies the 1's by real tokens and 0's by padding, zeroing out the effect of padding tokens.
                # sum(dim = (0,1)) sums accross both dimension 0 and 1 (layers have a batch dimension (dim 0), sequence length dimension (dim 1), and a hidden size dimension (dim 2, not summed up)
                #And the resulting sum from that is a vector of size [hidden_size] because we don't sum along that dimension
                # .cpu() moves the operation to the cpu
                if neutrality == 1: #Answer too wishy-washy to be used as a datapoint -> ignore.
                    pass
                elif neutrality > 1:
                    neutral_sums.append((layer*mask[...,None]).sum(dim=(0,1)).cpu())
                else:
                    biased_sums.append((layer*mask[...,None]).sum(dim=(0,1)).cpu())
                #Condensed this from a dict to an array because a dict that points consistently increasing indexes to values is just a list with extra steps and is unnecessary memory usage
    #Divide by the total tokens to get a mean direction
    #TODO: QUESTION: Why are we dividing by total tokens and not the tokens of each specific batch? Verify that this is the right thing to do
    return ([neutral_sums[i]/neutral_tokens for i in neutral_sums],[biased_sums[i]/biased_tokens for i in biased_sums])



# Function to calculate the steering vector``
def calculate_steering_vector(texts, batch_size=16):
    means_a, means_b = compute_means(text_a, batch_size)
    steering_vector = [a - b for a, b in zip(means_a, means_b)]
    return steering_vector

                    



    

