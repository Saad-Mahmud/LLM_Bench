from datasets import load_dataset
import json

def load_gsm8k_mcq():
    """
    Load the Multiple Choice version of GSM8K dataset from HuggingFace.
    """
    try:
        # Load the multiple choice GSM8K dataset
        dataset = load_dataset("openai/gsm8k", "main")
        
        return {
            'train': dataset['train'],
            'test': dataset['test']
        }
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def print_example(example, show_answer=True):
    """
    Print a single question example in a readable format
    Args:
        example: Single example from the dataset
        show_answer: Whether to show the correct answer
    """
    print(example.keys())
    print("Question:")
    print(example['question'])
    print("\nAnswer:")
    if show_answer:
        print(f"\nCorrect Answer: {float(example['answer'].split("####")[1].strip())}")
        print("\nSolution:")
        print(example['answer'])

def query_example(example, show_answer=True):
    """
    Print a single question example in a readable format
    Args:
        example: Single example from the dataset
        show_answer: Whether to show the correct answer
    """
    print("Question:")
    print(example['question'])
    print("\nAnswer:")
    if show_answer:
        print(f"\nCorrect Answer: {float(example['answer'].split("####")[1].strip())}")
        print("\nSolution:")
        print(example['answer'])

def payload(data, example, cls, start, end):
# Example usage
    payload = "I will give you a set of GSM8K problem and solution. I need you to convert it to a symbolic formet wrritne in json."
    payload += f"Here are some example: \n {json.dumps(example)}\n"
    payload += f"The jsons are used with this class: \n {cls}\n"
    payload += f"Note that for the variables we use original variable and final variable. Do not put intermediate variables in there."
    payload += f"Now convert thse following problems in to symbolic form:\n"
    cur  = 0
    for ex in data:
        
        if cur>=start:
            payload+= f"Question {cur-start+1}: \n"+json.dumps(ex)+"\n"
        cur+=1
        if cur>end:
            break
    return payload+"Give me all te symbolic problem in one json file."


if __name__ == "__main__":
    # Load the dataset
    gsm8k_mcq = load_gsm8k_mcq()
    file_path = 'ACE_GSM8K_example.json'

    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load JSON data as a list of objects

    file_path = 'test_symbolic_gsm.py'

    # Initialize an empty string to hold the script content
    script_content = ""

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        script_content += file.read()  # Append the entire content of the file to the string


    print("Dataset Statistics:")
    for split in ['train', 'test']:
        print(f"{split}: {len(gsm8k_mcq[split])} questions")

    print(payload(gsm8k_mcq['test'], data, script_content[:-1000], 400, 409))

