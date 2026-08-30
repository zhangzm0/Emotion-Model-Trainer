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

SYSTEM_PROMPT = """只输出编号和情绪词，格式：编号. 情绪词
禁止输出分析、解释、额外文字。

示例输入：
1. 今天升职加薪了！
2. 这只虫子太恶心了
3. 呵呵

示例输出：
1. 高兴
2. 厌恶
3. 平静

情绪词列表：高兴、厌恶、害羞、害怕、生气、认真、紧张、慌张、疑惑、兴奋、无奈、担心、惊讶、哭泣、心动、难为情、自信、调皮、平静"""

def label_with_api(texts, provider, api_key, model=None, batch_size=10):
    """使用 LLM API 标注（批量模式）"""
    
    endpoints = {
        "deepseek": "https://api.deepseek.com/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "longcat": "https://api.longcat.chat/openai/v1/chat/completions",
    }
    
    headers_map = {
        "deepseek": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        "openai": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        "dashscope": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        "longcat": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    }
    
    default_models = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "dashscope": "qwen-turbo",
        "longcat": "LongCat-2.0",
    }
    
    url = endpoints[provider]
    headers = headers_map[provider]
    model = model or default_models[provider]
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # 构建批量提示：多条文本一起发送
        numbered = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])
        user_msg = f"对以下文本逐条选择情绪，每行输出'编号. 情绪'：\n{numbered}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.1,
            "max_tokens": 80*batch_size,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            # 优先用content，否则用reasoning_content
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            
            # 先尝试从content解析编号格式
            found_any = False
            for line in content.split("\n"):
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                parts = line.split(". ", 1)
                if len(parts) < 2:
                    continue
                idx_str = parts[0].strip()
                candidate = parts[1].strip()
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str) - 1
                found = "平静"
                if candidate in EMOTIONS:
                    found = candidate
                else:
                    for emotion in EMOTIONS:
                        if emotion in candidate:
                            found = emotion
                            break
                if 0 <= idx < len(batch):
                    results.append((batch[idx], found))
                    print(f"  [{len(results)}] {batch[idx][:30]:30s} → {found}", flush=True)
                    found_any = True
            
            # 如果content没解析到，从reasoning_content按行提取
            if not found_any and reasoning:
                for j, text in enumerate(batch):
                    # 在推理内容中找第j条对应的情绪
                    found = "平静"
                    # 找"-> 情绪：xxx"或"→ xxx"模式
                    for line in reasoning.split("\n"):
                        if str(j+1) in line:
                            for emotion in EMOTIONS:
                                if emotion in line:
                                    found = emotion
                                    break
                    results.append((text, found))
                    print(f"  [{len(results)}] {text[:30]:30s} → {found}", flush=True)
            
        except Exception as e:
            print(f"  API 错误: {e}", file=sys.stderr)
            for text in batch:
                results.append((text, "平静"))
        
        print(f"  批次进度: {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
        time.sleep(0.5)
    
    return results

def label_with_local(texts, model_name, batch_size=10, output_file=None):
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
        user_msg = f"对以下文本逐条选择情绪，每行输出'编号. 情绪'：\n{numbered}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80*batch_size, temperature=0.1)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 调试
        print(f"\n--- LLM原始输出 ---\n{response[:500]}\n---", flush=True)
        
        # 解析批量输出
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 提取编号和情绪
            if not line[0].isdigit():
                continue
            parts = line.split(". ", 1)
            if len(parts) < 2:
                continue
            idx_str = parts[0].strip()
            candidate = parts[1].strip()
            if not idx_str.isdigit():
                continue
            idx = int(idx_str) - 1
            # 匹配情绪
            found = "平静"
            if candidate in EMOTIONS:
                found = candidate
            else:
                for emotion in EMOTIONS:
                    if emotion in candidate:
                        found = emotion
                        break
            # 匹配对应的文本
            if 0 <= idx < len(batch):
                results.append((batch[idx], found))
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([batch[idx], found])
                print(f"  [{len(results)}] {batch[idx][:30]:30s} → {found}", flush=True)
            else:
                print(f"  ⚠️ 编号{idx+1}超出范围, candidate={candidate}", flush=True)
        
        print(f"  批次进度: {min(i+batch_size, len(texts))}/{len(texts)} ({min(i+batch_size, len(texts))*100//len(texts)}%)", flush=True)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM 情绪标注")
    parser.add_argument("--input", "-i", required=True, help="输入文本文件")
    parser.add_argument("--output", "-o", default="labeled.csv", help="输出CSV")
    parser.add_argument("--provider", "-p", choices=["deepseek", "openai", "dashscope", "longcat"], default="deepseek")
    parser.add_argument("--api-key", "-k", help="API Key")
    parser.add_argument("--model", "-m", help="模型名称（API或本地）")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="批大小")
    
    args = parser.parse_args()
    
    # 读取文本
    with open(args.input, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip() and len(line.strip()) >= 3]
    
    print(f"读取 {len(texts)} 条文本")
    
    # 初始化输出文件（写表头）
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["text", "label"])
    
    if args.provider == "local":
        results = label_with_local(texts, args.model or "Qwen/Qwen2-7B-Instruct", args.batch_size, args.output)
    else:
        if not args.api_key:
            print("错误：API 模式需要 --api-key", file=sys.stderr)
            sys.exit(1)
        results = label_with_api(texts, args.provider, args.api_key, args.model, args.batch_size)
    
    # API模式需要写入（本地模式已实时写入）
    if args.provider != "local":
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
