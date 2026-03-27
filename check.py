import json

input_file = "./data/parkinson/dpo/parkinson_dpo_clean.jsonl"

print(f"开始检查文件: {input_file}\n" + "-"*50)

error_count = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        raw_line = line.strip()
        
        # 1. 检查是否是空行
        if not raw_line:
            print(f"❌ [第 {line_num} 行] 错误：这是一个空行。")
            error_count += 1
            continue
            
        # 2. 检查是否是 Markdown 标记
        if raw_line.startswith('```'):
            print(f"❌ [第 {line_num} 行] 错误：包含 Markdown 标记 '{raw_line}'。")
            error_count += 1
            continue
            
        # 3. 尝试解析 JSON
        try:
            json.loads(raw_line)
        except json.JSONDecodeError as e:
            print(f"❌ [第 {line_num} 行] JSON 格式破裂：{e}")
            print(f"   内容片段: {raw_line[:100]}...") # 只打印前100个字符防刷屏
            error_count += 1

print("-" * 50)
if error_count == 0:
    print("🎉 恭喜！文件格式完美，没有坏数据。")
else:
    print(f"总共发现 {error_count} 处问题，请根据行号去修改。")
