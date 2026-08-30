#!/usr/bin/env python3
"""
半自动数据标注脚本
用小模型对中文短句进行情绪标注，输出高置信度的标注结果

用法：
  python label_data.py --input texts.txt --output labeled.csv --threshold 0.85

输入：每行一条中文短句的文本文件
输出：CSV 格式 text,label
"""

import argparse
import csv
import sys
import numpy as np

# 19类情绪标签
EMOTIONS = [
    "高兴", "厌恶", "害羞", "害怕", "生气", "认真", "紧张", "慌张",
    "疑惑", "兴奋", "无奈", "担心", "惊讶", "哭泣", "心动", "难为情", "自信", "调皮", "平静"
]

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def label_texts(model_dir, input_file, output_file, threshold, batch_size=32):
    import onnxruntime as ort
    
    # 加载模型和分词器
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(f"{model_dir}/model.onnx")
    input_name = session.get_inputs()[0].name
    mask_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    
    # 读取文本
    with open(input_file, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"读取 {len(texts)} 条文本，开始标注...")
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                       max_length=128, return_tensors="np")
        
        outputs = session.run([output_name], {
            input_name: enc["input_ids"].astype(np.int64),
            mask_name: enc["attention_mask"].astype(np.float32)
        })[0]
        
        for j, logits in enumerate(outputs):
            probs = softmax(logits)
            conf = np.max(probs)
            if conf >= threshold:
                label = EMOTIONS[np.argmax(probs)]
                results.append((batch[j], label, conf))
        
        if (i // batch_size) % 10 == 0:
            print(f"  进度: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    # 写入CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "confidence"])
        for text, label, conf in results:
            writer.writerow([text, label, f"{conf:.4f}"])
    
    print(f"\n标注完成！")
    print(f"  输入: {len(texts)} 条")
    print(f"  高置信度(>={threshold}): {len(results)} 条 ({len(results)/len(texts)*100:.1f}%)")
    
    # 统计各类别
    from collections import Counter
    label_counts = Counter(r[1] for r in results)
    print(f"\n各类别分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文情绪半自动标注")
    parser.add_argument("--input", "-i", required=True, help="输入文本文件（每行一条）")
    parser.add_argument("--output", "-o", default="labeled.csv", help="输出CSV文件")
    parser.add_argument("--model", "-m", default="./emotion_model_19emo", help="模型目录")
    parser.add_argument("--threshold", "-t", type=float, default=0.85, help="置信度阈值")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="批大小")
    
    args = parser.parse_args()
    label_texts(args.model, args.input, args.output, args.threshold, args.batch_size)
