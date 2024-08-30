import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 指定设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)  # 将模型移动到指定设备
print("cao")

# 在推理时使用指定设备
@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    try:
        # 编码输入文本并放到指定设备
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

        # 使用模型生成文本
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
