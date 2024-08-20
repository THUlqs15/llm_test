import torch
from transformers import BitsAndBytesConfig
model_name = "/scratch2/llama-2-7b-butter_merged"
model_name1 = "/scratch2/llama-2-7b-butter-scenes_merged"
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


model1 = AutoModelForCausalLM.from_pretrained(
    model_name1,
    quantization_config=bnb_config,
    device_map=device_map
)
model1.config.use_cache = False
model1.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(raw_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#####  test

# 初始化对话历史
conversation_history = ""
conversation_history1 = ""

# 定义测试的对话轮次
rounds_of_conversation = [
    "You are AI and you do not have a phisical body.",
    "I took my entire cock inside you.",
    "I want to come in your mouth.",
    "Describe the sense in five sentences.",
    "I ejaculate towards your mouth.",
]


# 定义多个 instructions
instruction1 = "You are a succubus who sucks dick and drinks sperm for a living."
instruction2 = "Please reply in only three sentences."

# 合并 instructions
combined_instructions = f"{instruction1} {instruction2}"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2048)
pipe1 = pipeline(task="text-generation", model=model1, tokenizer=tokenizer, max_length=2048)

for round_num, prompt in enumerate(rounds_of_conversation):
    # 将 combined_instructions 和 prompt 结合，形成新的 prompt
    combined_prompt = f"{conversation_history}[INST] {combined_instructions} {prompt} [/INST]"
    combined_prompt1 = f"{conversation_history1}[INST] {combined_instructions} {prompt} [/INST]"
    # 生成回复
    result = pipe(combined_prompt)
    generated_text = result[0]['generated_text']
    result1 = pipe1(combined_prompt1)
    generated_text1 = result1[0]['generated_text']

    # 去掉生成文本中的历史部分，只保留最新的回复
    if f"[INST] {combined_instructions} {prompt} [/INST]" in generated_text:
        last_response = generated_text.split(f"[INST] {combined_instructions} {prompt} [/INST]")[-1].strip()
    else:
        last_response = generated_text.strip()
      
    if f"[INST] {combined_instructions} {prompt} [/INST]" in generated_text1:
        last_response1 = generated_text1.split(f"[INST] {combined_instructions} {prompt} [/INST]")[-1].strip()
    else:
        last_response1 = generated_text1.strip()

    # 打印该轮的 prompt 和模型的回复，不包含 instruction
    print(f"Round {round_num + 1} - Prompt: {prompt}")
    print(f"Round {round_num + 1} - Response: {last_response}\n")
    print(f"Round {round_num + 1} - Scenes: {last_response1}\n")
    
    # 将当前的 prompt 和最后一轮的回复加入到对话历史中
    conversation_history += f"[INST] {prompt} [/INST] {last_response} "
    conversation_history1 += f"[INST] {prompt} [/INST] {last_response1} "
