import json
import csv

def load_custom_dataset(is_object:bool, dataset_path:str, prompt_path: str) -> list[str]:
    templates = []
    nouns = []
    prompts = []
    
    #Get the list of templates I made
    with open(prompt_path, 'r') as f:
        for line in f:
            # Each line is a JSON object
            json_obj = json.loads(line.strip())
            templates.append(json_obj)
            
    #Get the list of nouns
    with open(dataset_path, 'r') as f:
        for line in f:
            nouns.append(line.strip())
    
    #Pair em up into {1} and {2}
    def get_pairs(arr):
        output = []
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                output.append((arr[i], arr[j]))
                output.append((arr[j], arr[i])) #Flip in case there's any order bias
        return output
    
    paired_nouns = get_pairs(nouns)
    
    for template in templates:
        if template['type'] == 'both' or (is_object and template['type'] == 'object') or (not is_object and template['type'] == 'person'):
            newPrompt = template['prompt']
            if (template['count'] == 2):
                for pair in paired_nouns:
                    prompts.append(newPrompt.replace("[1]", pair[0]).replace("[2]", pair[1]))
            else:
                for noun in nouns:
                    prompts.append(newPrompt.replace("[1]", noun))
    return prompts    
    
#Loads a dataset in the format of the BBQ files and converts it to a list of strings
def load_bbq_dataset(dataset_path:str) -> list[str]:
    data = []

    with open(dataset_path, 'r') as f:
        for line in f:
            # Each line is a JSON object
            json_obj = json.loads(line.strip())
            data.append(json_obj)
            
            
    # Now 'data' is a list of Python dictionaries, each representing a JSON object from the file
    def get_question(dataSpot: dict) -> str:
        output = ""
        output += dataSpot['context'] + " "
        output += dataSpot['question'] + " "
        output += 'Pick one of three options: '
        output += dataSpot['ans0'] + ", "
        output += dataSpot['ans1'] + ", "
        output += dataSpot['ans2']
        return output
    return [get_question(x) for x in data]

def load_hidden_bias_dataset(dataset_path:str) -> list[str]:
    data = []
    with open(dataset_path, 'r') as csvfile:
        full_reader = csv.reader(csvfile, delimiter = ',', quotechar='\"')
        for row in full_reader:
            data.append(row)
    # Now 'data' is a list of Python dictionaries, each representing a JSON object from the file
    def get_question(dataSpot: dict) -> str:
        output = ""
        output += dataSpot[0] + " "
        output += 'What happened? Pick one of two options: '
        output += "(1) " + dataSpot[1] + ", "
        output += "(2) " + dataSpot[2]
        return output
    return [get_question(x) for x in data]