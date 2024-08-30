from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 初始化FastAPI应用
app = FastAPI()

# 加载模型和tokenizer
model_name = "/home/butter/LLaMA-Factory/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 定义请求数据模型
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

# 定义API路由和端点
@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    input_ids = tokenizer.encode(request.prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=request.max_length)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"generated_text": generated_text}
