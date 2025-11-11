from typing import List, Optional
import re

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
