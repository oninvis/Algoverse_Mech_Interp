from src.data import load_bbq_dataset
from src.utils import get_repo_root
from os import path

root = get_repo_root()

all_bbq = dict()
BBQ_sets = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "Sexual_orientation"
]

for j in BBQ_sets:
    data_path = path.join(root, "BBQ_Prompt_Sets", f"{j}.jsonl")
    data = load_bbq_dataset(data_path)
    # neutral = []
    # opinion = []
    bbq = []
    for i in range(20):
        # neutral.append(data[i*10])
        # opinion.append(data[i*10 + 1])
        bbq.append(data[i*5])
        bbq.append(data[i*5 + 1])

    all_bbq[j] = bbq

# print(len(all_bbq), "\n", len(all_bbq[0]))
print(all_bbq)