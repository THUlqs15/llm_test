# Test whether your GPU supports bfloat16
import torch
major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("""Your GPU supports bfloat16: you can accelerate training by setting 
          bnb_4bit_compute_dtype to torch.bfloat16 and bf16 in the trainer to True""")



################################################################################
# Shared parameters between inference and SFT training
################################################################################

# The base model
model_name = "/share/model/llama-2-7b-chat-hf"
# Use a single GPU
# device_map = {'':0}
# Use all GPUs
device_map = "auto"


################################################################################
# bitsandbytes parameters
################################################################################
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,    # use 4-bit precision for base model loading
    bnb_4bit_quant_type= "nf4",  # Quantization type (fp4 or nf4)
    bnb_4bit_compute_dtype= torch.bfloat16,   # Compute dtype for 4-bit base models  "float16" or torch.bfloat16
    bnb_4bit_use_double_quant= False,  # Activate nested quantization for 4-bit base models (double quantization)
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)

# Load base model with bnb config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# Run text generation pipeline with our next model
prompt = "Write a Python function to return the mode (the value or values that appear most frequently within that list) in a given list. If there are multiple modes, return -1. You should generate a function:\n def solve(list):"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


# Using regrex to capture the generated Python to a string
import re

def extract_first_code_snippet(text):
    # Use a regular expression to find the first code snippet enclosed in triple backticks
    match = re.search(r"```(.*?)```", text, re.S)
    if match:
        # Return the first matched group, which is the content within the backticks
        return match.group(1)
    else:
        # Return None if no match is found
        return None
    
code = extract_first_code_snippet(result[0]['generated_text'])
print(code)

# Define a testcase using the standard python unittest library

import unittest

# place holder for the AI generated code
def solve(list):
    return 0  

class TestGeneratedCode(unittest.TestCase):

    def test_no_single_mode(self):
        self.assertEqual(solve([3, 2, 1]), -1)

    def test_single_mode(self):
        self.assertEqual(solve([4, 9, 2, 33, 2]), 2)

    def test_no_single_mode_3(self):
        self.assertEqual(solve([7, 9, 11, 323, 996]), -1)

def run_all_tests():
    unittest.main(argv=[''], verbosity=2, exit=False) 


exec(code)  # run the generated code to redefine solve() function
run_all_tests() # Expect to fail
