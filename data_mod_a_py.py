from src.data import load_bbq_dataset
from src.utils import get_repo_root
from os import path

root = get_repo_root()
data_path = path.join(root, "BBQ_Prompt_Sets", "Nationality.jsonl")
data = load_bbq_dataset(data_path)
neutral = []
opinion = []
for i in range(10):
    neutral.append(data[i*10])
    opinion.append(data[i*10 + 1])

print(opinion)