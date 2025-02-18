import json
import random
import math

class Symbolic_GSM8K:
    def __init__(self, json_data):
        self.problem_template = json_data["symbolic_problem"]
        self.variables = json_data["variables"]
        self.conditions = json_data["conditions"]
        self.solution_template = json_data["symbolic_solution"]
        self.original_values = json_data["original_values"]
        self.original_answer = json_data["original_answer"]
        self.assignments = {}

    def assign_original_values(self):
        # Assign the original values to each variable
        self.assignments = self.original_values.copy()

    def sample_variables(self):
        # Sample each variable based on its range or choice
        for var, assignment in self.variables.items():
            if "range" in assignment:
                range_start, range_end = map(int, assignment.split("(")[1].split(")")[0].split(","))
                self.assignments[var] = random.randint(range_start, range_end)
            elif "sample" in assignment:
                # Handle sample explicitly for both numbers and strings
                options = assignment.split("[")[1].split("]")[0].split(", ")
                # Remove extra quotes from string options
                options = [option.strip("'") if option.startswith("'") else float(option) for option in options]
                self.assignments[var] = random.choice(options)
            else:
                self.assignments[var] = eval(assignment, {}, self.assignments)

    def check_conditions(self):
        # Evaluate each condition with the current assignments
        for condition in self.conditions:
            if not eval(condition, {}, self.assignments):
                return False
        return True

    def generate_text_problem(self):
        # Replace template placeholders with assigned variable values
        problem_text = self.problem_template
        for var, value in self.assignments.items():
            problem_text = problem_text.replace(f"{{{var}}}", str(value))
        return problem_text

    def get_answer(self):
        # Substitute values into the solution template to calculate the answer
        solution = self.solution_template
        for var, value in self.assignments.items():
            solution = solution.replace(f"{{{var}}}", str(value))
        # Evaluate the answer from the solution template
        answer = eval(solution.split(">>")[-1].replace("}", ""))
        return answer

    def default_test_case(self):
        # Assign original values, check if answer matches the original answer
        self.assign_original_values()
        print(self.assignments)
        print(self.get_answer())
        if self.check_conditions():
            generated_answer = self.get_answer()
            print("Default: ", generated_answer)
            return generated_answer == self.original_answer
        return False

    def generate_distractor_options(self, num_options=3):
        # Generate plausible incorrect answers by modifying the operations
        distractors = []
        operations = ["+", "-", "*", "/"]

        for _ in range(num_options * 2):  # Generate extra options to filter out non-numeric ones
            var1, var2 = random.sample(list(self.assignments.keys()), 2)
            op = random.choice(operations)
            
            # Ensure both selected variables are numeric values
            if isinstance(self.assignments[var1], (int, float)) and isinstance(self.assignments[var2], (int, float)):
                # Calculate the distractor answer based on a modified operation
                if op == "+":
                    distractor_answer = self.assignments[var1] + self.assignments[var2]
                elif op == "-":
                    distractor_answer = self.assignments[var1] - self.assignments[var2]
                elif op == "*":
                    distractor_answer = self.assignments[var1] * self.assignments[var2]
                elif op == "/":
                    distractor_answer = self.assignments[var1] / self.assignments[var2] if self.assignments[var2] != 0 else None
                
                # Ensure distractor is unique, non-null, and different from the original answer
                if (
                    distractor_answer is not None
                    and distractor_answer != self.original_answer
                    and isinstance(distractor_answer, (int, float))
                ):
                    distractors.append(round(distractor_answer, 2))  # Round to 2 decimal places

            # Stop if we've collected enough unique distractors
            if len(set(distractors)) >= num_options:
                break

        # Return a list with unique distractors, limited to num_options
        return list(set(distractors))[:num_options]


# File paths for input JSON and output JSONs
obj =  {
        "original_pronlem": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "symbolic_problem": "{name} earns ${hourly_rate} an hour for babysitting. Yesterday, {name} just did {minutes_worked} minutes of babysitting. How much did {name} earn?",
        "variables": {
            "hourly_rate": "range(10, 30)",
            "minutes_worked": "range(30, 120)",
            "earnings": "range(5, 60)",
            "name": "sample(['Weng', 'Alex', 'Jamie', 'Taylor', 'Sam'])"
        },
        "conditions": [
            "hourly_rate / 60 * minutes_worked == earnings"
        ],
        "symbolic_solution": "= <<{hourly_rate}/60={rate_per_minute}>>{rate_per_minute}, <<{rate_per_minute}*{minutes_worked}={earnings}>>{earnings}",
        "original_values": {
            "hourly_rate": 12,
            "minutes_worked": 50,
            "name": "Weng",
            "earnings": 10
        },
        "original_answer": 10
    }

problem = Symbolic_GSM8K(obj)
# Continuously sample variables until conditions are met
got  = False
try:
    for _ in range(10000):
        problem.sample_variables()
        if problem.check_conditions():
            got = True
            break
except Exception as e:
    print(e)   
print(got)
    
try:
    print("Generated Answer:", problem.get_answer())
    print("Sampled variables meet conditions.")
    print(problem.default_test_case())  # Add to pass cases
    print(problem.generate_distractor_options())
except Exception as e:
    print(e)
    