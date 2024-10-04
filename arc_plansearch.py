import os
import requests
import time
import itertools
import ast
import json
from dotenv import load_dotenv
import sys
import datetime

load_dotenv()

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def extract_data(response, data_specs):
    prompt = ("Given the following text and specification, extract and return the data -- and only the data -- in the format specified:\n"
    f"Text: {response}\n"
    f"Specification: {data_specs}"
    )
    try:
        data = get_response(prompt, "openai/gpt-4o")
    except Exception as e:
        print(f"Error in extract_data: {e}")
        return ""
    return data

def get_response(prompt, model):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    print("Prompt: ", prompt)
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openrouter_api_key}",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                 headers=headers, 
                                 json=payload)
        time.sleep(1.5)  # sleep to not overwhelm open router qps limits
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data['choices'][0]['message']['content'].strip()
        print(f"{model}: {response_text}")
        return response_text
    except Exception as e:
        print(f"Error in get_response: {e}")
        return ""

def generate_observations_1(problem, model):
    prompt = f"""Here are a few examlpes of input-output grid pairs where the objective is to derive a function that can transform each input_grid to the corresponding output_grid. Return several useful, non-obvious, and correct observations about the problem, like hints to solve the problem. Do NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.

Problem: {problem["train"]}

Observations:"""
    response = get_response(prompt, model)
    observations = extract_data(response, "The given text contains observations about a programming problem. I want each observation as a complete string in it's entirety in an array [observation1, observation2, ...]")
    return json.loads(observations)

def generate_observations_2(problem, observations_1, model):
    prompt = (
        "You are an expert Python programmer. Here are a few examlpes of input-output"
        "grid pairs where the objective is to derive a function that can transform "
        "each input_grid to the corresponding output_grid. Following that are several correct observations about the problem. "
        "You will brainstorm several new, useful, and correct observations about the problem, derived from the given observations. "
        "You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.\n\n"
        "Problem: {}\n\n"
        "Initial observations:\n"
        "{}\n\n"
        "New observations:"
    ).format(problem["train"], '\n'.join(observations_1))

    response = get_response(prompt, model)
    observations = extract_data(response, "The given text contains observations about a programming problem. I want each observation as a complete string in it's entirety in an array [observation1, observation2, ...]")
    return json.loads(observations)

def generate_plans_1(problem, observations, model):
    plans = []
    for subset in itertools.chain.from_iterable(itertools.combinations(observations, r) for r in range(1, len(observations) + 1)):
        prompt = (
            "Here is the competitive programming problem:\n"
            "{}\n\n"
            "Here are the intelligent observations to help solve the problem:\n"
            "{}\n\n"
            "Use these observations above to brainstorm a natural language solution / plan to the problem above. "
            "Note that your intuition may lead you astray, so come up with simple, creative ideas that go beyond "
            "what you would usually come up with and exceeds your narrow intuition. Quote relevant parts of the "
            "observations EXACTLY before each step of the solution. QUOTING IS CRUCIAL."
        ).format(problem["train"], '\n'.join(subset))
        
        response = get_response(prompt, model)
        plan = extract_data(response, "The given text contains a natural language solution plan. I want the plan as a single string.")
        plans.append(plan)
    return plans

def generate_plans_2(problem, plans, model):
    alternative_plans = []
    for plan in plans:
        prompt = f"""Here are a few examlpes of input-output grid pairs where the objective is to derive a function that can transform each input_grid to the corresponding output_grid:
{problem["train"]}

The following is a proposed solution to the problem:
{plan}

Please critique this solution and provide an alternative approach that addresses any potential weaknesses or explores a different strategy."""
        response = get_response(prompt, model)
        alternative_plan = extract_data(response, "The given text contains a natural language solution plan. I want the plan as a single string.")
        alternative_plans.append(alternative_plan)
    return alternative_plans

def generate_pseudocodes(plans, model):
    pseudocodes = []
    for plan in plans:
        prompt = f"""You are an expert programmer. Please convert the following natural language solution into detailed pseudocode:
{plan}"""
        response = get_response(prompt, model)
        pseudocode = extract_data(response, "The given text is pseudocode. I want the pseudocode as a single string.")
        pseudocodes.append(pseudocode)
    return pseudocodes

def generate_codes(problem, pseudocodes, model):
    codes = []
    for pseudocode in pseudocodes:
        prompt = f"""You are an expert Python programmer. Here are a few examlpes of input-output grid pairs where the objective is to derive a function that can transform each input_grid to the corresponding output_grid. Following that is the pseudocode that describes how to solve the problem. You will generate a correct Python program that matches said specification and pseudocode and passes all tests. You will NOT return anything except for the program inside markdown codeblocks.

Problem: {problem["train"]}

Pseudocode:
{pseudocode}

Python code:"""
        response = get_response(prompt, model)
        code = extract_data(response, "The given text is Python code. I want the code as a single executable string: preserving indents and line breaks, without any markdown formatting.")
        codes.append(code)
    return codes

def plansearch(problem, model):
    # inject diversity
    observations_1 = generate_observations_1(problem, model)
    observations_2 = generate_observations_2(problem, observations_1, model)
    all_observations = observations_1 + observations_2
    
    plans_1 = generate_plans_1(problem, all_observations, model)
    plans_2 = generate_plans_2(problem, plans_1, model)
    all_plans = plans_1 + plans_2
    
    # ensure diversity
    pseudocodes = generate_pseudocodes(all_plans, model)
    
    # get samples
    codes = generate_codes(problem, pseudocodes, model)
    
    return codes

def execute_solution(solution_code, problem, test=False):
    # Parse the solution code into an AST
    tree = ast.parse(solution_code)
    
    # Find the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            break
    else:
        raise ValueError("No function definition found in the solution code")
    
    # Compile and execute the solution code
    exec(compile(tree, '<string>', 'exec'))
    
    # Get the function from the local namespace
    solution_function = locals()[function_name]
    
    # Call the function with the example inputs as the arguments
    if test:
        output = solution_function(problem["test"][0]["input"])
        if output != problem["test"][0]["output"]:
            return False
        return True

    for example in problem["train"]:
        output = solution_function(example["input"])
        if output != example["output"]:
            return False
    return True


def get_passing(solutions, problem):
    passing = []
    for solution in solutions:
        try:
            passing = [solution for solution in solutions if execute_solution(solution, problem)]
        except Exception as e:
            print(f"Solution failed: {e}")
    return passing

def test(passing, problem):
    correct = []
    try:
        correct = [solution for solution in passing if execute_solution(solution, problem, test=True)]
    except Exception as e:
        print(f"Solution failed: {e}")
    return correct

def get_problems_training():
    random_sample = ["8d510a79.json", "6455b5f5.json", "4612dd53.json", "c0f76784.json", "3906de3d.json", "a61f2674.json", "0b148d64.json", "b1948b0a.json", "868de0fa.json", "1190e5a7.json", "780d0b14.json", "1b60fb0c.json", "f9012d9b.json", "73251a56.json", "0520fde7.json", "4c4377d9.json", "6e19193c.json", "6aa20dc0.json", "bda2d7a6.json", "2281f1f4.json"]
    problems = []
    for file in random_sample:
        with open(f"data/training/{file}", "r") as f:
            problems.append(json.loads(f.read()))
    return problems

def get_problems_eval():
    random_sample = ["c1990cce.json", "4f537728.json", "d19f7514.json", "917bccba.json", "69889d6e.json", "94414823.json", "aa4ec2a5.json", "1d0a4b61.json", "14754a24.json", "7e02026e.json", "b7cb93ac.json", "e74e1818.json", "ba9d41b8.json", "33b52de3.json", "ac605cbb.json", "0b17323b.json", "2072aba6.json", "a680ac02.json", "af22c60d.json", "1c0d0a4b.json"]
    problems = []
    for file in random_sample:
        with open(f"data/evaluation/{file}", "r") as f:
            problems.append(json.loads(f.read()))
    return problems

if __name__ == "__main__":
    # Create log directory if it doesn't exist
    log_dir = "attempts/arc_plansearch"
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with current date and time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/arc_plansearch_log_{current_time}.txt"
    
    # Set up logging
    sys.stdout = Logger(log_filename)

    models = ["openai/o1-mini", "openai/o1-preview"]
    datasets = ["training", "evaluation"]
    problems_attempted = 0
    max_attempts = 1
    for dataset in datasets:
        print(f"\n--- Running on {dataset.capitalize()} Set ---")
        
        # Get problems for the current dataset
        problems = []
        if dataset == "training":
            problems = get_problems_training()
        else:
            problems = get_problems_eval()

        # Initialize metrics for each model
        model_metrics = {model: {
            "total_problems": 0,
            "total_solutions": 0,
            "total_passing": 0,
            "total_correct": 0,
            "score": 0,
            "avg_passing_over_correct": 0
        } for model in models}

        for problem in problems:
            for model in models:
                solutions = plansearch(problem, model)
                passing = get_passing(solutions)
                correct = test(passing, problem)
                
                # Update metrics for this model
                model_metrics[model]["total_problems"] += 1
                model_metrics[model]["total_solutions"] += len(solutions)
                model_metrics[model]["total_passing"] += len(passing)
                model_metrics[model]["total_correct"] += len(correct)
                
                if len(correct) > 0:
                    model_metrics[model]["score"] += 1
                    model_metrics[model]["avg_passing_over_correct"] += len(passing) / len(correct)
                
                print(f"Problem {problem} with model {model}: {len(solutions)} solutions, {len(passing)} passing, {len(correct)} correct.")
                problems_attempted += 1
                if problems_attempted >= max_attempts:
                    break

        # Calculate final metrics and print results for each model
        for model, metrics in model_metrics.items():
            total_problems = metrics["total_problems"]
            metrics["avg_passing_over_correct"] /= total_problems if total_problems > 0 else 1
            
            print(f"\nMetrics for {model} on {dataset} set:")
            print(f"Total problems: {total_problems}")
            print(f"Total solutions: {metrics['total_solutions']}")
            print(f"Total passing: {metrics['total_passing']}")
            print(f"Total correct: {metrics['total_correct']}")
            print(f"Score: {metrics['score']}/{total_problems}")
            print(f"Average passing over correct: {metrics['avg_passing_over_correct']:.2f}")