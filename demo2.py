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




from transformers import TrainingArguments

################################################################################
# Model name and directories
################################################################################

# The base model
model_name = "/share/model/llama-2-7b-chat-hf"
# The instruction dataset to use
dataset_name = "./Formatted_Dataset"
#dataset_name = "/scratch2/py18k"
# Fine-tuned model name
new_model = "/scratch2/llama-2-7b-butter"
# Output directory where the model predictions and checkpoints will be stored
output_dir = "/scratch2/results"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1
bias="none"
task_type="CAUSAL_LM"

################################################################################
# Training parameters (passed to TrainingArguments)
################################################################################

# Number of training epochs
num_train_epochs = 1
# Number of training steps (overrides num_train_epochs)
# max_steps = 100
# Enable fp16/bf16 training (set bf16 to True if supported by your GPU)
fp16 = False
bf16 = True
# Batch size per GPU for training
per_device_train_batch_size = 4
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine"
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 0

################################################################################
# Monitoring parameters
################################################################################

# Logging dir (for tensorboard)
logging_dir = f"{output_dir}/logs"
# Log every X updates steps
logging_steps = 25
# Monitoring and Visualizing tools
report_to = "tensorboard"

################################################################################
# SFTTrainer parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 512
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False


# Load dataset
from datasets import load_from_disk
dataset = load_from_disk(dataset_name)
print(dataset[0])


from peft import PeftModel
from trl import SFTTrainer

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    #max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    logging_steps=logging_steps,
    logging_dir=logging_dir,
    report_to=report_to, 
)


from peft import LoraConfig
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias=bias,
    task_type=task_type,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)






# # Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)


# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
merged_model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Save merged model to disk (optional)
merged_model.save_pretrained(f'{new_model}_merged')


prompt = "My dic is so big:"
pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=512)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


#exec(code)
run_all_tests() # Expect to pass
