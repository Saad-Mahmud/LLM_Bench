from GSM_agents.gsm8k_sympolic import Symbolic_GSM8K
import random
import json
from tqdm import tqdm

class BaseGSMAgent(object):

    def __init__(self, data_file = 'dataset/ACE_GSM8K.json', result_file = 'output_accuracy_log.txt'):
        self.data_file = data_file
        self.result_file = result_file
        try:
            with open(self.data_file, 'r') as file:
                data = json.load(file)
                self.DATA = [obj for obj in data]
            file.close()
        except Exception as e:
            print(f"Error while loading data file: {e}")
        
        
    def make_options(self, problem, obj, answer, n_options = 6):
        # Create options and shuffle
            class_names = [ f"{answer}"]
            if n_options>1:
                class_names.append(f"{obj["original_answer"]}")
            if n_options>2:
                class_names.append("0")
            if n_options>3:
                class_names.append("None of the options are correct.")
            if n_options>4:
                choice  = random.sample([answer-i for i in range(-100,101,1) if (i!=0 or abs(answer-i-obj["original_answer"])<2)], n_options-4)
                for i in choice:
                    class_names.append(f"{i}")
            random.shuffle(class_names)

            
            correct_answer = ""
            for class_idx, ans in enumerate(class_names):
                if ans == str(answer):
                    correct_answer = f"Option {chr(ord('A')+class_idx)}"
            assert len(correct_answer) > 6
            return class_names, correct_answer

    def log(self, info):
        with open(self.result_file, 'a') as file:
            file.write(info + '\n')

    def test(self, llm_agent, n = 200, n_options = 6, verbos = True, n_turns = 3):
        self.ST(llm_agent, n=n, n_options=n_options, verbos=verbos)
        
    def ST(self, llm_agent, n = 200, n_options = 6, verbos = True):
        n= max( min(200, n), 1)
        test_data = random.sample(self.DATA, n)
        success = 0
        total_ans = 0
        for obj in tqdm(test_data, desc="Test Running: "):
            problem = Symbolic_GSM8K(obj)            
            for _ in range(100000):
                problem.sample_variables()
                if problem.check_conditions() and problem.get_answer() != obj["original_answer"]:
                    break
            if not problem.check_conditions() or problem.get_answer() == obj["original_answer"]:
                continue

            prompt = (
                "Print the correct option to this math problem. Do not print anythin else."
                " Finish your answer by printing the tag <|end|>. \nQuestion: "
                + problem.generate_text_problem()
            )
            try:
                answer = float(problem.get_answer())
            except Exception:
                continue
            
            options, correct_answer = self.make_options(problem, obj, answer, n_options)
            

            # Append options to prompt
            for idx, option in enumerate(options):
                prompt += f"\nOption {chr(ord('A')+idx)}: {option}"

            if verbos:
                print("Prompt: ", prompt)
                print("Correct Answer: ",correct_answer)

            response = llm_agent.answer(prompt, [f"Option {chr(ord('A')+idx)}" for idx in range(n_options)])
            if response.status_code == 200:
                data = response.json()
                if verbos:
                    print("Output:", data["output"])
                    print("logits:", data["logits"])
                    print("Time taken:", data["time_taken"])
                if data["output"] == correct_answer:
                    success += 1
                total_ans +=1
            else:
                continue
            
            if total_ans > 0:
                accuracy = success / total_ans
                if verbos:
                    print(f"Answered:  {total_ans}, Accuracy: {accuracy}")
                self.log(f"{llm_agent.name} answered:  {total_ans}, its accuracy: {accuracy}")  

    