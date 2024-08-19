

import json
from datasets import Dataset, DatasetDict

def format_data(data):
    formatted_data = []
    
    for entry in data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")
        
        # Combine instruction and input into human_text
        human_text = f"""{instruction} The given input is {input_text}."""
        
        # Set output as assistant_text
        assistant_text = output_text
        
        # Create the formatted string
        formatted_text = f'<s>[INST] {human_text} [/INST] ```{assistant_text}``` </s>'
        
        # Add the formatted text to the list
        formatted_data.append({'text': formatted_text})
    
    return formatted_data

def process_json_file(input_file):
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Remove "system" and "history" fields
    for entry in data:
        entry.pop("system", None)
        entry.pop("history", None)
    
    # Format the data
    formatted_data = format_data(data)
    
    # Create a Dataset object from the formatted data
    #dataset = Dataset.from_dict({'text': formatted_data})
    formatted_data = [{'text': str(item['text'])} for item in formatted_data]
    dataset = Dataset.from_dict({'text': [item['text'] for item in formatted_data]})
    
    return dataset

# File paths
input_file = './Conversation_ONLY.json'
output_dir = './Formatted_Dataset'

# Process the JSON file and create the Dataset
dataset = process_json_file(input_file)

# Optionally, shuffle the dataset
dataset = dataset.shuffle(seed=42)

# Save the dataset to disk
dataset.save_to_disk(output_dir)
