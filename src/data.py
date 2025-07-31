import json

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