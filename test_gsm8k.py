from datasets import load_dataset
import random
import requests

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

# Example usage
if __name__ == "__main__":
    # Load the dataset
    gsm8k_mcq = load_gsm8k_mcq()
    
    if gsm8k_mcq:
        # Print dataset statistics
        print("Dataset Statistics:")
        for split in ['train', 'test']:
            print(f"{split}: {len(gsm8k_mcq[split])} questions")
        
        # Print a random example
        #print("\nRandom example from training set:")
        #random_example = random.choice(gsm8k_mcq['train'])
        #print_example(random_example, show_answer=True)
        crr = 0
        total = 0
        for example in gsm8k_mcq['test']:
            #print_example(example, show_answer=True)

            prompt = "Start by reasoning about the math problem step-by-step. Think about how you would write a python code to solve it. Consider the variabls you gonna use. " 
            prompt += "Finally print the correct option. Keep it short. Finish your answer by printing the tag <END> \n Question: "+example['question']
            try:
                answer = int(example['answer'].split("####")[1].strip())
            except Exception as e:
                continue
            class_names = [
                f"{answer}",
                "0.0",
                f"{answer-random.randint(0,10)-1}",
                f"{answer+random.randint(0,10)+1}",
                "None of the options are correct."
            ]
            random.shuffle(class_names)

            ma = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
            correct_answer = ""
            for i, ans in enumerate(class_names):
                if ans == str(answer):
                    correct_answer = f"Option {ma[i]}"
                    if random.random()<0.0:
                        class_names[i] = "None of the options are correct." 
            assert len(correct_answer)>6

            prompt = prompt + f"\nOption A. {class_names[0]}"
            prompt = prompt + f"\nOption B. {class_names[1]}"
            prompt = prompt + f"\nOption C. {class_names[2]}"
            prompt = prompt + f"\nOption D. {class_names[3]}"
            prompt = prompt + f"\nOption E. {class_names[4]}"
            
            query = {
                "prompt": prompt,
                "type": "mcq",
                "class_names": ["Option A", "Option B", "Option C", "Option D", "Option E"],
                "cot": True,
                "eof_tag": "<END>"
            }
            print(class_names)
            print(correct_answer)
            response = requests.post("http://localhost:8000/generate", json=query)
            if response.status_code == 200:
                data = response.json()
                print("Output:", data["output"])
                #print("Class_names:", data["class_names"])
                crr += (data["output"][0] == correct_answer)
                total +=1 
                print("Time taken:", data["time_taken"])
            else:
                print("Error:", response.text)
            
            if total > 0 :
                print("Accuracy: ", total, crr/total)
            if total==200:
                break
