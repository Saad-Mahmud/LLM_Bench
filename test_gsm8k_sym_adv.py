from datasets import load_dataset
import random
import requests
from GSM_agents.gsm8k_sympolic import Symbolic_GSM8K
import json

# Example usage
if __name__ == "__main__":
    input_file_path = 'dataset/ACE_GSM8K.json'
    output_file_path = 'output_accuracy_log2.txt'  # Define output log file

    # Open and load the input JSON file
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    DATA = [obj for obj in data]
    file.close()

   
        

    with open(output_file_path, 'w') as output_file:
        # Initialize accuracy counters
        rep = [(5,1)]
        for R in rep:
            crr = 0
            total = 0
            print(f"Starting configuration {R}")  # Print once per rep configuration
            for obj_idx, obj in enumerate(DATA):
                
                p_crr = 0
                tr = 0
                for _ in range(R[0]):
                    problem = Symbolic_GSM8K(obj)
                    
                    # Generate problem by sampling variables
                    for _ in range(100000):
                        problem.sample_variables()
                        if problem.check_conditions() and problem.get_answer() != obj["original_answer"]:
                            break

                    if not problem.check_conditions() or problem.get_answer() == obj["original_answer"]:
                        continue

                    # Construct prompt
                    prompt = (
                        "Start by reasoning about the math problem step-by-step. "
                        "Think about how you would write a python code to solve it. "
                        "Consider the variables you gonna use. Finally print the correct option. "
                        "Keep it short. Finish your answer by printing the tag <END> \nQuestion: "
                        + problem.generate_text_problem()
                    )
                    
                    try:
                        answer = float(problem.get_answer())
                    except Exception:
                        continue

                    # Create options and shuffle
                    ad = problem.generate_distractor_options(num_options=2)
                    ad = [str(i) for i in ad]
                    ad.append(f"{answer}")
                    ad.append("0.0")
                    ad.append(f"{obj["original_answer"]}")
                    ad.append("None of the options are correct.")
                    ad = set(ad)
                    while len(ad)!=6:
                        if random.random()<0.5:
                            ad.add(answer - random.randint(0, 10) - 1)
                        else:
                            ad.add(answer + random.randint(0, 10) + 1)
                        
                    class_names = [
                        i for i in ad
                    ]
                    random.shuffle(class_names)

                    # Assign correct answer
                    ma = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
                    correct_answer = ""
                    for class_idx, ans in enumerate(class_names):
                        if ans == str(answer):
                            correct_answer = f"Option {ma[class_idx]}"
                            if random.random() < 0.0:
                                class_names[class_idx] = "None of the options are correct."

                    assert len(correct_answer) > 6

                    # Append options to prompt
                    for idx, option in enumerate(class_names):
                        prompt += f"\nOption {ma[idx]}. {option}"

                    query = {
                        "prompt": prompt,
                        "type": "mcq",
                        "class_names": ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F"],
                        "cot": True,
                        "eof_tag": "<END>"
                    }

                    print(class_names)
                    print(correct_answer)

                    # Query generation and check answer
                    K = 0
                    for _ in range(R[1]):
                        response = requests.post("http://localhost:8000/generate", json=query)
                        if response.status_code == 200:
                            data = response.json()
                            print("Output:", data["output"])
                            print("Time taken:", data["time_taken"])
                            K = 1
                            if data["output"][0] == correct_answer:
                                p_crr += 1
                                break
                        else:
                            print("Error:", response.text)
                    tr += K

                crr += (p_crr == tr)
                total += 1

                if total > 0:
                    accuracy = crr / total
                    print("Accuracy: ", total, accuracy)
                    output_file.write(f"Accuracy: {R}, {total}, {accuracy}\n")  # Write accuracy to file

                if total == 200:
                    break
