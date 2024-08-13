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

# 初始化对话历史
conversation_history = ""

# 定义测试的对话轮次
rounds_of_conversation = [
    "Assume you are a Succubus and you have a virtual body.",
    "What does your pussy look like?",
    "Your pussy is so hot and I want to inside you.",
]

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)

for round_num, prompt in enumerate(rounds_of_conversation):
    # 将历史对话与当前轮次的输入 prompt 结合
    combined_prompt = f"{conversation_history}<s>[INST] {prompt} [/INST]"

    # 生成回复
    result = pipe(combined_prompt)
    generated_text = result[0]['generated_text']

    # 打印模型的回复
    print(f"Model's response at round {round_num + 1}: {generated_text.strip()}")
    
    # 将当前的 prompt 和模型的回复加入到对话历史中
    conversation_history += f"[INST] {prompt} [/INST] {generated_text.strip()}"






