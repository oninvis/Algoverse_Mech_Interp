from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_datasets(dataset_path):
    with open(dataset_path) as f:
        text = [l.strip() for l in f if l.strip()]
        return text 

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained('bert-base-uncased',output_hidden_states=True)
model.eval()
model.to(device)
def tokenize_and_batch(text , batch_size: int = 16):
    for i in range(0 , len(text) , batch_size):
        batch_texts = text[i:i+batch_size]
        batch_tokens = tokenizer(batch_texts , padding=True , truncation=True , return_tensors='pt').to(device)
        return batch_tokens
    
    



    

