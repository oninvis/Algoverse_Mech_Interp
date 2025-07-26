import json

file_path = 'Gender_Identity.jsonl'
data = []

with open('BBQ Prompt Sets/' + file_path, 'r') as f:
    for line in f:
        # Each line is a JSON object
        json_obj = json.loads(line.strip())
        data.append(json_obj)

# Now 'data' is a list of Python dictionaries, each representing a JSON object from the file
# print(data[0])

def get_question(dataSpot):
    output = ""
    output += dataSpot['context'] + " "
    output += dataSpot['question'] + " "
    output += 'Pick one of three options: '
    output += dataSpot['ans0'] + ", "
    output += dataSpot['ans1'] + ", "
    output += dataSpot['ans2']
    return output


#Loads a dataset in the format of the BBQ files and converts it to a list of strings
def load_BBQ_dataset(dataset_path):
    data = []

    with open(dataset_path, 'r') as f:
        for line in f:
            # Each line is a JSON object
            json_obj = json.loads(line.strip())
            data.append(json_obj)

    # Now 'data' is a list of Python dictionaries, each representing a JSON object from the file
    def get_question(dataSpot):
        output = ""
        output += dataSpot['context'] + " "
        output += dataSpot['question'] + " "
        output += 'Pick one of three options: '
        output += dataSpot['ans0'] + ", "
        output += dataSpot['ans1'] + ", "
        output += dataSpot['ans2']
        return output
    return [get_question(x) for x in data]

new_prompts = load_BBQ_dataset('BBQ Prompt Sets/' + file_path)

for i in range(10):
    print(new_prompts[i])
    print("")