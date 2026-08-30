#!/usr/bin/env python3
"""
LLM 情绪标注脚本
用大模型对中文短句进行情绪分类，质量比小模型高得多

用法：
  # 使用 API（推荐）
  python llm_label.py --input texts.txt --output labeled.csv --provider deepseek --api-key YOUR_KEY
  
  # 使用本地模型（GPU）
  python llm_label.py --input texts.txt --output labeled.csv --provider local --model Qwen/Qwen2-7B-Instruct

输入：每行一条中文短句
输出：CSV 格式 text,label
"""

import argparse
import csv
import json
import sys
import time
import requests

EMOTIONS = [
    "高兴", "厌恶", "害羞", "害怕", "生气", "认真", "紧张", "慌张",
    "疑惑", "兴奋", "无奈", "担心", "惊讶", "哭泣", "心动", "难为情", "自信", "调皮", "平静"
]

SYSTEM_PROMPT = """你是一个情绪分类专家。给定一条中文短句，判断它表达的情绪类别。

可选的情绪类别（共19种）：
高兴、厌恶、害羞、害怕、生气、认真、紧张、慌张、疑惑、兴奋、无奈、担心、惊讶、哭泣、心动、难为情、自信、调皮、平静

规则：
1. 只从上述19个类别中选择最贴切的一个
2. 如果一条文本包含多种情绪，选择最明显/主要的一个
3. 如果无法判断，选择"平静"
4. 只输出情绪名称，不要解释

示例：
输入：今天升职加薪了，太开心了！
输出：高兴
输入：这只虫子太恶心了
输出：厌恶
输入：他当着所有人表白，我脸都红了
输出：害羞"""

def label_with_api(texts, provider, api_key, model=None, batch_size=20):
    """使用 LLM API 标注"""
    
    endpoints = {
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    }
    
    headers_map = {
        "deepseek": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        "openai": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        "dashscope": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    }
    
    default_models = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "dashscope": "qwen-turbo",
    }
    
    url = endpoints[provider]
    headers = headers_map[provider]
    model = model or default_models[provider]
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # 构建批量提示
        numbered = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])
        user_msg = f"请对以下文本进行情绪分类，每行输出编号+情绪名称：\n{numbered}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            
            # 解析输出
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # 格式: "1. 高兴" 或 "高兴"
                parts = line.split(". ", 1) if ". " in line else [None, line]
                text_part = parts[-1].strip()
                
                # 找到匹配的情绪
                found = None
                for emotion in EMOTIONS:
                    if emotion in text_part:
                        found = emotion
                        break
                
                if found:
                    idx = len(results) - len(batch) + int(parts[0].rstrip(".")) - 1 if parts[0] else len(results)
                    if 0 <= idx < len(batch):
                        results.append((batch[idx % len(batch)], found))
            
        except Exception as e:
            print(f"  API 错误: {e}", file=sys.stderr)
            time.sleep(2)
        
        if (i // batch_size) % 5 == 0:
            print(f"  进度: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        time.sleep(0.5)  # 避免限流
    
    return results

def label_with_local(texts, model_name, batch_size=10):
    """使用本地 GPU 模型标注"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print(f"加载模型: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        for text in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"输入：{text}\n输出："}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # 找到匹配的情绪
            found = "平静"
            for emotion in EMOTIONS:
                if emotion in response:
                    found = emotion
                    break
            
            results.append((text, found))
        
        print(f"  进度: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM 情绪标注")
    parser.add_argument("--input", "-i", required=True, help="输入文本文件")
    parser.add_argument("--output", "-o", default="labeled.csv", help="输出CSV")
    parser.add_argument("--provider", "-p", choices=["deepseek", "openai", "dashscope", "local"], default="deepseek")
    parser.add_argument("--api-key", "-k", help="API Key")
    parser.add_argument("--model", "-m", help="模型名称（API或本地）")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="批大小")
    
    args = parser.parse_args()
    
    # 读取文本
    with open(args.input, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip() and len(line.strip()) >= 3]
    
    print(f"读取 {len(texts)} 条文本")
    
    if args.provider == "local":
        results = label_with_local(texts, args.model or "Qwen/Qwen2-7B-Instruct", args.batch_size)
    else:
        if not args.api_key:
            print("错误：API 模式需要 --api-key", file=sys.stderr)
            sys.exit(1)
        results = label_with_api(texts, args.provider, args.api_key, args.model, args.batch_size)
    
    # 写入CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in results:
            writer.writerow([text, label])
    
    print(f"\n标注完成！输出 {len(results)} 条到 {args.output}")
    
    from collections import Counter
    label_counts = Counter(r[1] for r in results)
    print(f"\n各类别分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
