from datasets import load_dataset

# Load the dataset
# dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca')
dataset = load_dataset('/share/data/python_code_instructions_18k_alpaca')

dataset = dataset['train']
print(dataset[0])
