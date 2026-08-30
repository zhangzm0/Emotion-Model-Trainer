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

SYSTEM_PROMPT = """你是情绪分类专家。对每条中文短句，先分析情绪线索，再从以下19种情绪中选择最贴切的一个：

高兴、厌恶、害羞、害怕、生气、认真、紧张、慌张、疑惑、兴奋、无奈、担心、惊讶、哭泣、心动、难为情、自信、调皮、平静

输出格式（每条一行）：
分析：xxx 情绪：xxx"""

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
        
        # 逐条发送（带思考的逐条分析更准确）
        for j, text in enumerate(batch):
            user_msg = f"输入：{text}\n输出："
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            }
            
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                
                # 解析 "分析：xxx\n情绪：xxx" 格式
                found = None
                analysis = ""
                has_emotion_line = False
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("分析：") or line.startswith("分析:"):
                        analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    if line.startswith("情绪：") or line.startswith("情绪:"):
                        has_emotion_line = True
                        candidate = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                        # 精确匹配优先
                        if candidate in EMOTIONS:
                            found = candidate
                        else:
                            # 模糊匹配
                            for emotion in EMOTIONS:
                                if emotion in candidate:
                                    found = emotion
                                    break
                
                # 如果没有"情绪："行，在整个输出中找情绪词（排除分析行）
                if not has_emotion_line:
                    for emotion in EMOTIONS:
                        if emotion in content:
                            found = emotion
                            break
                
                found = found or "平静"
                results.append((text, found))
                # 实时输出，方便监控质量
                print(f"  [{len(results)}] {text[:25]:25s} → {found:4s} | {analysis[:30]}")
                
            except Exception as e:
                print(f"  API 错误: {e}", file=sys.stderr)
                results.append((text, "平静"))
                time.sleep(2)
            
            time.sleep(0.3)  # 避免限流
        
        print(f"  批次进度: {min(i+len(batch), len(texts))}/{len(texts)}")
    
    return results

def label_with_local(texts, model_name, batch_size=10):
    """使用本地 GPU 模型标注（批量模式）"""
    import torch
    
    # 优先用 ModelScope 下载（国内快）
    try:
        from modelscope import AutoTokenizer, AutoModelForCausalLM
        print(f"通过 ModelScope 加载: {model_name}...")
    except ImportError:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"通过 HuggingFace 加载: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # 构建批量提示：多条文本一起发送
        numbered = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])
        user_msg = f"对以下文本逐条分析情绪，每行输出'分析：xxx 情绪：xxx'：\n{numbered}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80*batch_size, temperature=0.1)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 解析批量输出
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 找情绪
            found = "平静"
            if "情绪：" in line or "情绪:" in line:
                candidate = line.split("情绪：")[-1].split("情绪:")[-1].strip()
                for emotion in EMOTIONS:
                    if emotion in candidate:
                        found = emotion
                        break
            # 匹配对应的文本
            for j, text in enumerate(batch):
                if text in line or (j < len(batch) and str(j+1) in line):
                    results.append((text, found))
                    break
        
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
