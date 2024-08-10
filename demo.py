from datasets import load_dataset

# Load the dataset
# dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca')
dataset = load_dataset('/share/data/python_code_instructions_18k_alpaca')

dataset = dataset['train']
print(dataset[0])

# Define a function to transform the data
def format_py18k(example):

    instruction = example['instruction']
    instr_input = example['input']
    output = example['output']

    human_text = f"""{instruction} The given input is {instr_input}."""
    assistant_text = output
    # Apply the new template
    reformatted_entry = {'text': f'<s>[INST] {human_text} [/INST] ```{assistant_text}``` </s>'}

    return reformatted_entry


# Apply the transformation
formatted = dataset.map(format_py18k)
formatted = formatted.remove_columns(['instruction', 'input', 'output', 'prompt'])
formatted = formatted.shuffle(seed=42)

print(formatted[0])

formatted.save_to_disk('/scratch2/py18k')
