import os
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "DS")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "DS_clean")

def clean_dataset(input_file, output_file):
    # Phrases we want to filter out to avoid human-like personas
    human_phrases = [
        r'\bmy husband\b',
        r'\bmy wife\b',
        r'\bmy kids\b',
        r'\bmy children\b',
        r'\bmy mom\b',
        r'\bmy dad\b',
        r'\bmy sister\b',
        r'\bmy brother\b',
        r'\bmy childhood\b',
        r'\bgrowing up\b',
        r'\bwhen I was in\b',
        r'\bin highschool\b',
        r'\bcollege degree\b',
        r'\bmy hair\b',
        r'\bmy legs\b',
        r'\bmy eyes\b',
        r'\bI hated\b',
        r'\bI loved\b',
        r'\bI was terrified when\b',
        r'\bI remember\b'
    ]
    
    pattern = re.compile('|'.join(human_phrases), re.IGNORECASE)
    
    total = 0
    kept = 0
    removed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                assistant_text = data.get('assistant', '')
                
                total += 1
                
                if pattern.search(assistant_text):
                    removed += 1
                else:
                    kept += 1
                    f_out.write(line)
                    
            except json.JSONDecodeError:
                pass
                
    return total, kept, removed

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
        in_path = os.path.join(DATA_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(in_path):
            print(f"Processing {filename}...")
            total, kept, removed = clean_dataset(in_path, out_path)
            print(f"  Total: {total}")
            print(f"  Kept: {kept} ({(kept/total)*100 if total > 0 else 0:.2f}%)")
            print(f"  Removed: {removed} ({(removed/total)*100 if total > 0 else 0:.2f}%)")
            print()
        else:
            print(f"File not found: {in_path}")

if __name__ == '__main__':
    main()
