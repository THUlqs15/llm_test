import torch
from transformers import BitsAndBytesConfig
major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("""Your GPU supports bfloat16: you can accelerate training by setting 
          bnb_4bit_compute_dtype to torch.bfloat16 and bf16 in the trainer to True""")


model_name = "/scratch2/llama-2-7b-butter"


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

prompt = "I fuck you pussy hardly:"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
