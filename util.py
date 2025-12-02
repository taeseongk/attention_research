from typing import List, Optional
import re
import json
from pathlib import Path
import argparse

def clean_context(context: str) -> str:
    clean = re.sub(r'<[^>]+>', '', context)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def split_into_sentences(context: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def key_sentence_prompt(question: str, context: str) -> str:
    return f"""A question and a passage are shown below. Select the key sentence in the passage that supports to answer the question correctly. Only output the exactly same sentence from the passage without other additional words.

Question: {question}

Passage: {context}

Answer:"""

def answer_prompt(question: str, context: List[str], method: str, key_sentence: Optional[List[str]] = None) -> str:
    prompt = ""

    if method == "baseline" or method == "autopasta":
        prompt = f"""Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words.
        
Context: {" ".join(context)}
        
Question: {question}
        
Answer:"""
    elif method == "iter_prompt" and key_sentence:
        prompt = f"""Answer the question below, paired with a context that provides background knowledge, and a key sentence. Only output the answer without other context words.
            
Context: {" ".join(context)}

Key Sentence: {" ".join(key_sentence)}
        
Question: {question}
        
Answer:"""
    return prompt

def apply_template(prompt: str, tokenizer, add_gen_prompt: bool):
    messages = [{"role": "user", "content": prompt}] 
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_gen_prompt
    )
    if not add_gen_prompt:
        text = text[:-len("<|eot_id|>")]
    return text

def get_differences():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, required=True, help="Path to first file")
    parser.add_argument("--file2", type=str, required=True, help="Path to second file")
    args = parser.parse_args()

    if args.file1== args.file2:
        print(f"Error: Same files")
        return

    path1 = Path(args.file1)
    path2 = Path(args.file2)

    if not path1.exists():
        print(f"Error: File1 not found")
        return
    if not path2.exists():
        print(f"Error: File2 not found")

    with open(path1, 'r') as f:
        data1 = json.load(f)
    with open(path2, 'r') as f:
        data2 = json.load(f)

    results1 = sorted(data1["results"], key=lambda x: x['id'])
    results2 = sorted(data2["results"], key=lambda x: x['id'])

    for item1, item2 in zip(results1, results2):
        if item1['exact_score'] != item2['exact_score'] or item1['f1_score'] != item2['f1_score']:
            print(item1["question"])
            print(f"{item1['prediction']}\nEM: {item1['exact_score']}\nF1: {item1['f1_score']}\n")
            print(f"{item2['prediction']}\nEM: {item2['exact_score']}\nF1: {item2['f1_score']}")
            input()

if __name__ == "__main__":
    main()
