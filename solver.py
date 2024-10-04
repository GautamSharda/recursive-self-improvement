import os
import json
import requests
from dotenv import load_dotenv
from retry import retry
import re
from datetime import datetime
import threading
from queue import Queue
import time
import openai  # Change this line

# Load environment variables from .env file
load_dotenv()

def load_tasks(directory):
    tasks = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                task_data = json.load(f)
                if task_data:  # Check if the file is not empty
                    tasks.append((filename, task_data))
    return tasks

class InvalidResponseError(Exception):
    pass

def is_valid_2d_grid(response):
    try:
        grid = json.loads(response)
        return (isinstance(grid, list) and
                all(isinstance(row, list) and
                    all(isinstance(item, int) for item in row)
                    for row in grid))
    except json.JSONDecodeError:
        return False

def process_task(task_name, task_data, file_handle, model, first_task=False, last_task=False):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    train_data = task_data['train']
    test_data = task_data['test']
    
    task_description = f"Task: {task_name}\n\nTrain examples:\n"
    for i, example in enumerate(train_data):
        task_description += f"Example {i+1}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
    
    task_description += f"Test input:\n{test_data[0]['input']}\n\nBased on the training examples, provide the output for the test input. The output should be a 2D grid of integers. Only provide the grid, without any additional text or explanation."

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openrouter_api_key}",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": task_description}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                 headers=headers, 
                                 json=payload)
        time.sleep(1.5)  # sleep to not overwhelm open router qps limits
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data['choices'][0]['message']['content'].strip()

        if is_valid_2d_grid(response_text):
            print(f"Task: {task_name}")
            print(f"Claude's response:\n{response_text}")
            
            expected_output = task_data['test'][0]['output']
            is_correct = json.loads(response_text) == expected_output
            
            result = {
                'task_name': task_name,
                'model': model,
                'response': json.loads(response_text),
                'expected': expected_output,
                'is_correct': is_correct
            }

            # Write only the JSON result to the file
            json.dump(result, file_handle)
            file_handle.write("\n")  # Add a newline for separation

            # Write current date and time and task description for the first task
            if first_task:
                current_time = datetime.now().isoformat()
                task_info = {
                    "timestamp": current_time,
                    "task_description": task_description
                }
                json.dump(task_info, file_handle)
                file_handle.write("\n")  # Add a newline for separation

            print(f"Correct: {is_correct}")
            print("---")
        else:
            raise InvalidResponseError(f"Response is not a valid 2D grid of integers: {response_text}")
    
    except (requests.RequestException, InvalidResponseError) as e:
        print(f"Error processing task {task_name} with model {model}: {str(e)}")
        error_result = {
            'task_name': task_name,
            'model': model,
            'response': f"Error: {str(e)}",
            'expected': task_data['test'][0]['output'],
            'is_correct': False
        }
        
        # Write only the error result to the file
        json.dump(error_result, file_handle)
        file_handle.write("\n")  # Add a newline for separation

    except Exception as e:
        print(f"Unexpected error processing task {task_name} with model {model}: {str(e)}")

    # Write current date and time for the last task
    if last_task:
        current_time = datetime.now().isoformat()
        file_handle.write(f'{{"timestamp": "{current_time}"}}\n')

@retry(exceptions=(requests.RequestException,), tries=3, delay=1, backoff=2)
def process_task(task_name, task_data, file_handle, model, first_task=False):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    train_data = task_data['train']
    test_data = task_data['test']
    
    task_description = f"Task: {task_name}\n\nTrain examples:\n"
    for i, example in enumerate(train_data):
        task_description += f"Example {i+1}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
    
    task_description += f"Test input:\n{test_data[0]['input']}\n\nBased on the training examples, provide the output for the test input. The output should be a 2D grid of integers. Only provide the grid, without any additional text or explanation."

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openrouter_api_key}",
        }

        payload = {
            "model": model,  # Use the model passed as an argument
            "messages": [
                {"role": "user", "content": task_description}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                 headers=headers, 
                                 json=payload)
        time.sleep(1.5) # sleep to not overwhelm open router qps limits
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data['choices'][0]['message']['content'].strip()

        if is_valid_2d_grid(response_text):
            print(f"Task: {task_name}")
            print(f"Claude's response:\n{response_text}")
            
            # Check if response matches expected output
            expected_output = task_data['test'][0]['output']
            is_correct = json.loads(response_text) == expected_output
            
            result = {
                'task_name': task_name,
                'model': model,  # Include the model in the result
                'response': json.loads(response_text),
                'expected': expected_output,
                'is_correct': is_correct
            }

            # Write result to file using json.dump for proper formatting
            file_handle.write("    {\n")
            json.dump(result, file_handle, indent=4)  # Properly format the JSON
            file_handle.write(",\n")
            file_handle.flush()  # Ensure the data is written immediately

            if first_task:
                file_handle.seek(0)
                file_handle.write("# Example Prompt\n")
                file_handle.write(f"'''\n{task_description}\n'''\n\n")
                file_handle.write("results = [\n")
                file_handle.flush()
                file_handle.seek(0, 2)  # Move to the end of the file

            print(f"Correct: {is_correct}")
            print("---")
        else:
            raise InvalidResponseError(f"Response is not a valid 2D grid of integers: {response_text}")
    
    except (requests.RequestException, InvalidResponseError) as e:
        print(f"Error processing task {task_name} with model {model}: {str(e)}")
        error_result = {
            'task_name': task_name,
            'model': model,
            'response': f"Error: {str(e)}",
            'expected': task_data['test'][0]['output'],
            'is_correct': False
        }
        
        # Write error result to file using json.dump for proper formatting
        file_handle.write("    {\n")
        json.dump(error_result, file_handle, indent=4)  # Properly format the JSON
        file_handle.write(",\n")
        file_handle.flush()

    except Exception as e:
        print(f"Unexpected error processing task {task_name} with model {model}: {str(e)}")
        # Handle unexpected errors similarly to above

def get_available_models():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
    }
    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
    response.raise_for_status()
    models = response.json()
    return [model['id'] for model in models['data']]

def generate_report(attempts_dir):
    report = {}
    
    # Iterate through all files in the attempts directory
    for filename in os.listdir(attempts_dir):
        if filename.startswith("attempt-") and filename.endswith(".py"):
            model_name = filename[8:-3].replace("-", "/")  # Extract model name from filename
            filepath = os.path.join(attempts_dir, filename)
            
            correct_count = 0
            total_count = 0
            
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.strip():  # Check if the line is not empty
                            result = json.loads(line.strip())  # Load each JSON object
                            if result.get('is_correct'):
                                correct_count += 1
                            total_count += 1
            
            except Exception as e:  # Handle errors in reading/loading
                print(f"Error reading/loading report for {model_name}: {str(e)}")
                correct_count = 0
                total_count = 0
            
            report[model_name] = {
                'correct': correct_count,
                'total': total_count,
                'accuracy': correct_count / total_count if total_count > 0 else 0
            }
    
    # Write the report to a file
    report_path = os.path.join(attempts_dir, "report.py")
    with open(report_path, 'w') as f:
        f.write("report = {\n")
        for model, stats in report.items():
            f.write(f"    '{model}': {{\n")
            f.write(f"        'correct': {stats['correct']},\n")
            f.write(f"        'total': {stats['total']},\n")
            f.write(f"        'accuracy': {stats['accuracy']:.4f}\n")
            f.write("    },\n")
        f.write("}\n")
    
    print(f"Report generated: {report_path}")

def get_report(report_path):
    with open(report_path, 'r') as f:
        content = f.read()
        report_dict = eval(content.split('=', 1)[1].strip())
    
    # Sort models by accuracy in descending order
    sorted_models = sorted(report_dict.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("Model Performance Report (Best to Worst):")
    print("----------------------------------------")
    for model, stats in sorted_models:
        print(f"Model: {model}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Total: {stats['total']}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print("----------------------------------------")

def process_model(model, tasks, attempts_dir, result_queue):
    model_filename = f"attempt-{model.replace('/', '-')}.py"
    model_filepath = os.path.join(attempts_dir, model_filename)

    with open(model_filepath, 'w+') as f:
        start_time = datetime.now()
        f.write(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("results = [\n")
        for index, task in enumerate(tasks):
            task_name, task_data = task
            try:
                process_task(task_name, task_data, f, model, first_task=(index == 0), last_task=(index == len(tasks) - 1))
            except Exception as e:
                print(f"Failed to process task {task_name} with model {model}: {str(e)}")
                # Continue with the next task
        f.write("]\n\n")
        end_time = datetime.now()
        f.write(f"# End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        duration = end_time - start_time
        f.write(f"# Total duration: {duration}\n")
    
    result_queue.put(f"Completed processing for model: {model}")

def main(models, experiment_name=None):
    if not models:
        available_models = get_available_models()
        error_message = "The models list is empty. Please provide at least one model.\n\nAvailable models:\n"
        error_message += "\n".join(available_models)
        raise ValueError(error_message)

    evaluation_dir = 'data/training'
    tasks = load_tasks(evaluation_dir)
    print(f"Processing {len(tasks)} tasks for {len(models)} models.")
    
    if experiment_name:
        attempts_dir = os.path.join('attempts', experiment_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempts_dir = os.path.join('attempts', f"attempt-{timestamp}")
    
    os.makedirs(attempts_dir, exist_ok=True)

    threads = []
    result_queue = Queue()

    for model in models:
        thread = threading.Thread(target=process_model, args=(model, tasks, attempts_dir, result_queue))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print results from the queue
    while not result_queue.empty():
        print(result_queue.get())

    # Generate report after all models have been processed
    generate_report(attempts_dir)

    # Print the report
    report_path = os.path.join(attempts_dir, "report.py")
    get_report(report_path)

def o1_test():
    print("Starting O1 test...")
    evaluation_dir = 'data/training'
    tasks = load_tasks(evaluation_dir)
    print(f"Loaded {len(tasks)} tasks from {evaluation_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempts_dir = os.path.join('attempts', f"o1-test-{timestamp}")
    os.makedirs(attempts_dir, exist_ok=True)
    print(f"Created attempts directory: {attempts_dir}")
    
    output_file = os.path.join(attempts_dir, "attempt-openai-o1-preview.py")
    print(f"Output will be saved to: {output_file}")
    
    openai.api_key = os.getenv("OPENAI_API_KEY")  # Add this line
    
    with open(output_file, 'w') as f:
        start_time = datetime.now()
        f.write(f"# Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("results = [\n")
        
        for i, (task_name, task_data) in enumerate(tasks, 1):
            print(f"\nProcessing task {i}/{len(tasks)}: {task_name}")
            try:
                train_data = task_data['train']
                test_data = task_data['test']
                
                task_description = f"Task: {task_name}\n\nTrain examples:\n"
                for i, example in enumerate(train_data):
                    task_description += f"Example {i+1}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
                
                task_description += f"Test input:\n{test_data[0]['input']}\n\nBased on the training examples, provide the output for the test input. The output should be a 2D grid of integers. Only provide the grid, without any additional text or explanation."

                print("Sending request to OpenAI API...")
                completion = openai.ChatCompletion.create(  # Change this line
                    model="o1-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": task_description}
                    ]
                )
                print("Received response from OpenAI API")
                
                response_text = completion.choices[0].message.content.strip()
                
                if is_valid_2d_grid(response_text):
                    expected_output = task_data['test'][0]['output']
                    is_correct = json.loads(response_text) == expected_output
                    
                    result = {
                        'task_name': task_name,
                        'model': "openai/o1-preview",
                        'response': json.loads(response_text),
                        'expected': expected_output,
                        'is_correct': is_correct
                    }
                    
                    json.dump(result, f)
                    f.write(",\n")
                    f.flush()
                    
                    print(f"Task: {task_name}")
                    print(f"Correct: {is_correct}")
                    print("---")
                else:
                    print(f"Invalid response received: {response_text}")
                    raise InvalidResponseError(f"Response is not a valid 2D grid of integers: {response_text}")
                
            except Exception as e:
                print(f"Error processing task {task_name}: {str(e)}")
                error_result = {
                    'task_name': task_name,
                    'model': "openai/o1-preview",
                    'response': f"Error: {str(e)}",
                    'expected': task_data['test'][0]['output'],
                    'is_correct': False
                }
                json.dump(error_result, f)
                f.write(",\n")
                f.flush()
            
            print("Waiting 1 second before next task...")
            time.sleep(1)  # To avoid rate limiting
        
        f.write("]\n\n")
        end_time = datetime.now()
        f.write(f"# End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        duration = end_time - start_time
        f.write(f"# Total duration: {duration}\n")
    
    print(f"\nO1 test completed. Results saved to: {output_file}")
    print(f"Total duration: {duration}")

def get_modified_task_content_list():
    task_list = [
        "8d510a79.json",
        "6455b5f5.json",
        "4612dd53.json",
        "c0f76784.json",
        "3906de3d.json",
        "a61f2674.json",
        "0b148d64.json",
        "b1948b0a.json",
        "868de0fa.json",
        "1190e5a7.json",
        "780d0b14.json",
        "1b60fb0c.json",
        "f9012d9b.json",
        "73251a56.json",
        "0520fde7.json",
        "4c4377d9.json",
        "6e19193c.json",
        "6aa20dc0.json",
        "bda2d7a6.json",
        "2281f1f4.json"
    ]
    task_content_list = []
    for task_file in task_list:
        file_path = os.path.join(training_folder, task_file)
        with open(file_path, 'r') as f:
            content = f.read()
            task_content_list.append(content)
    
    modified_task_content_list = []
    for content in task_content_list:
        last_output_index = content.rfind('"output"')
        if last_output_index != -1:
            second_last_bracket = content.rfind(']', 0, content.rfind(']'))
            if second_last_bracket != -1:
                modified_content = content[:last_output_index + 8] + " <your_response>" + content[second_last_bracket + 1:]
                modified_task_content_list.append(modified_content)
    
    return modified_task_content_list

def get_prompt_list():
    modified_task_content_list = get_modified_task_content_list()

    prompt_list = []
    for modified_content in modified_task_content_list:
        prepended_content = "Here is another series of input-output pairs.\n\n" + modified_content
        appended_content = prepended_content + "\n\nReason about the problem first. Then reflect on your reasoning, and correct yourself if you find any mistakes. Then construct the output for the test input. Then reflect on your output, and correct yourself if you find any mistakes. Repeat as many times as necessary, and only once you are confident, provide a final output."
        prompt_list.append(appended_content)
    
    return prompt_list

if __name__ == "__main__":
    from model_list import models
    import sys
    import os

    training_folder = 'data/training'
    num_files = len([f for f in os.listdir(training_folder) if os.path.isfile(os.path.join(training_folder, f))])
    print(f"Number of files in {training_folder}: {num_files}")
    # Print the name of every file in sorted order
    print("Files in the training folder:")
    for filename in sorted(os.listdir(training_folder)):
        if os.path.isfile(os.path.join(training_folder, filename)):
            print(filename)
    
    print(get_prompt_list())
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else None
    # o1_test()
    # main(models, experiment_name)