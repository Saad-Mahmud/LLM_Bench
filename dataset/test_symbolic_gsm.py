import json
import random

class SymbolicProblem:
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
        if self.check_conditions():
            generated_answer = self.get_answer()
            return generated_answer == self.original_answer
        return False

# File paths for input JSON and output JSONs
input_file_path = 'ACE_GSM8K.json'
pass_file_path = 'temp.json'
bad_file_path = 'temp.json'
assert  input_file_path!= pass_file_path
assert  input_file_path!= bad_file_path

# Open and load the input JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)  # Load JSON data as a list of objects

# Initialize counters and lists for passing and failing cases
bad_count = 0
pass_cases = []
bad_cases = []

# Process each object in the array one by one
for i, obj in enumerate(data):
    problem = SymbolicProblem(obj)

    # Continuously sample variables until conditions are met
    got = False
    try:
        for _ in range(40000):
            problem.sample_variables()
            if problem.check_conditions():
                got = True
                break
    except Exception as e:
        pass   
    if not got:
        print(f"************** ISSSSS: {i}")
        bad_count += 1
        bad_cases.append(obj)  # Add to bad cases
        continue

    try:
        print("Generated Answer:", problem.get_answer())
        #print("Sampled variables meet conditions.")
        if not problem.default_test_case():
            bad_count += 1
            bad_cases.append(obj)  # Add to bad cases
        else:
            pass_cases.append(obj)  # Add to pass cases
    except Exception as e:
        print(f"************** ISSSSS: {i}") 
        print(e)
        bad_count += 1
        bad_cases.append(obj)  # Add to bad cases

# Write passing and failing cases to respective JSON files
with open(pass_file_path, 'w') as pass_file:
    json.dump(pass_cases, pass_file, indent=4)

with open(bad_file_path, 'w') as bad_file:
    json.dump(bad_cases, bad_file, indent=4)

print("Total bad cases:", bad_count)
print("Total pass cases:", len(pass_cases))
