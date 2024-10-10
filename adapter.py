from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

def load_model(model_name_or_path, adapter_name_or_path):
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载适配器模型
    model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    
    # 将模型放入评估模式
    model.eval()
    return tokenizer, model

def run_inference(model, tokenizer, history, max_new_tokens=50):
    # 将历史对话拼接为输入张量
    prompt = "\n".join(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    # 解码生成的张量为文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    model_name_or_path: "/root/lqs/LLaMA-Factory-main/llama3_models/models/Meta-Llama-3-8B-Instruct"
    adapter_name_or_path: "/root/lqs/LLaMA-Factory-main/llama3_models/9_11"
    
    # 加载模型和分词器
    tokenizer, model = load_model(model_name_or_path, adapter_name_or_path)
    
    # 初始化对话历史
    history = []
    
    print("Chat with the model (type 'exit' to end the conversation)")
    while True:
        # 获取用户输入
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        # 将用户输入添加到历史中
        history.append(f"You: {user_input}")
        
        # 运行推理
        generated_text = run_inference(model, tokenizer, history)
        
        # 提取模型生成的响应
        response = generated_text[len("\n".join(history)) + 1:]
        
        # 将响应添加到历史中
        history.append(f"Model: {response}")
        
        # 打印结果
        print(f"Model: {response}")

if __name__ == "__main__":
    main()
