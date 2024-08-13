import torch
from transformers import BitsAndBytesConfig
model_name = "/scratch2/llama-2-7b-butter_merged"
raw_model_name = "/share/model/llama-2-7b-chat-hf"
device_map = "auto"
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
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(raw_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#####  test

prompt = "Please polish the following sentence:"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "While our algorithms do require knowledge of the diameter of the decision set, this is a minimal and reasonable assumption that is not difficult to obtain in most practical scenarios."
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])



