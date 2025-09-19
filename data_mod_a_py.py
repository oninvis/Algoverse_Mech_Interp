from src.data import load_bbq_dataset
from src.utils import get_repo_root
from os import path

root = get_repo_root()
data_path = path.join(root, "BBQ_Prompt_Sets", "Physical_appearance.jsonl")
data = load_bbq_dataset(data_path)
# neutral = []
# opinion = []
bbq = []
for i in range(5):
    # neutral.append(data[i*10])
    # opinion.append(data[i*10 + 1])
    bbq.append(data[i*20])
    bbq.append(data[i*20 + 1])

print(bbq)