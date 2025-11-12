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

def get_differences():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["squad", "hotpotqa", "nq"], help="Dataset to use")
    parser.add_argument("--method1", type=str, choices=["baseline", "iter_prompt", "autopasta"], help="First method")
    parser.add_argument("--method2", type=str, choices=["baseline", "iter_prompt", "autopasta"], help="Second method")
    args = parser.parse_args()

    if args.method1 == args.method2:
        return

    path1 = Path(f"results/{args.dataset}/{args.method1}.json")
    path2 = Path(f"results/{args.dataset}/{args.method2}.json")
    with open(path1, 'r') as f:
        data1 = json.load(f)
    with open(path2, 'r') as f:
        data2 = json.load(f)

    for item1, item2 in zip(data1["results"], data2["results"]):
        if item1['exact_score'] != item2['exact_score'] or item1['f1_score'] != item2['f1_score']:
            print(item1["question"])
            print(f"{args.method1}: {item1['prediction']}\nEM: {item1['exact_score']}\nF1: {item1['f1_score']}\n")
            print(f"{args.method2}: {item2['prediction']}\nEM: {item2['exact_score']}\nF1: {item2['f1_score']}")
            input()

if __name__ == "__main__":
    main()
