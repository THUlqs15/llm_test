from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 定义请求体的模型
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0

# 初始化 FastAPI 应用
print("cao0")
app = FastAPI()

# 加载模型和分词器
model_path = "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("cao1")
model = AutoModelForCausalLM.from_pretrained(model_path)
print("cao2")
# 定义生成文本的 API
@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    try:
        # 编码输入文本
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(model.device)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
