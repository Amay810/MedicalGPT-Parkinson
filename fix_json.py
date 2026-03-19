import json

input_file = "./data/parkinson/sft/parkinson_sft_cot.jsonl"
output_file = "./data/parkinson/sft/parkinson_sft_cot_fixed.jsonl"

def get_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception:
        with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
            return f.readlines()

lines = get_lines(input_file)
valid_count = 0
invalid_count = 0

with open(output_file, 'w', encoding='utf-8') as fout:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "conversations" in data:
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                valid_count += 1
            else:
                invalid_count += 1
        except Exception:
            invalid_count += 1

print("Fix Completed!")
print("Valid lines: " + str(valid_count))
print("Invalid lines: " + str(invalid_count))
print("Output saved to: " + output_file)