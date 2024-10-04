import os

def print_file_names():
    print("Files in the evaluation folder:")
    evaluation_folder = 'data/evaluation'
    files = []
    for filename in sorted(os.listdir(evaluation_folder)):
        if os.path.isfile(os.path.join(evaluation_folder, filename)):
            files.append(filename)

    import random

    # Select 20 random files
    selected_files = random.sample(files, min(20, len(files)))

    print("Selected 20 random files:")
    for file in selected_files:
        print(file)

def api_check():
    import requests
    import time
    from dotenv import load_dotenv
    load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    print("Prompt: ", "hi")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openrouter_api_key}",
        }

        payload = {
            "model": "openai/o1-mini",
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                 headers=headers, 
                                 json=payload)
        time.sleep(1.5)  # sleep to not overwhelm open router qps limits
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data['choices'][0]['message']['content'].strip()
        print(f"openai/o1-mini: {response_text}")
        return response_text
    except Exception as e:
        print(f"Error in get_response: {e}")
        return ""

api_check()