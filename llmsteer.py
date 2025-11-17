import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from typing import List
from pathlib import Path
import copy
import argparse
import json
from tqdm import tqdm

from data import QADatasetLoader
from eval import QAEvaluator

@dataclass
class LLMSteerConfig:
    alpha: float
    top_k: int
    filter_special_tokens: bool = True
    prefix_prompt_1: str = """
Answer the question based on the given passages. Only give me the answer and do not output any other words. The following are given passages.
""".strip()
    prefix_prompt_2: str = """
Respond to the query using only the provided texts. Only return the answer. DO NOT return any extra words. Below are the provided texts.
""".strip()


class LLMSteer:
    """
    LLMSteer implementation from the paper.
    """

    def __init__(self, config: LLMSteerConfig, model_name: str):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.device = "cuda"

        self.attn_scores_1 = defaultdict(lambda: defaultdict(list))
        self.attn_scores_2 = defaultdict(lambda: defaultdict(list))
        self.ctxt_len = None
        self.sel_tokens = {}

        self.cached_kv = None
        self.cached_text = None
        self.original_cached_kv = None

        self.last_context = None

        
    def compute_steering_matrices(self, context: str):
        def is_special_or_whitespace(token_idx: int, input_ids: torch.Tensor) -> bool:
            """Check if token is special token or just whitespace."""
            if token_idx >= len(input_ids):
                return True
            
            token_id = input_ids[token_idx].item()
            
            if token_id in self.special_token_ids:
                return True
            
            token_text = self.tokenizer.decode([token_id])
            stripped = token_text.strip()

            if any(marker in token_text for marker in ['<|', '|>', 'user', 'assistant', 'system']):
                return True
            
            if len(stripped) == 0 or stripped in ['.', ',', '!', '?', ';', ':',  '\n', '\t']:
                return True
            
            return False

        def select_tokens(layer_idx: int, input_ids: torch.Tensor):
            all_scores_1 = torch.zeros(self.ctxt_len)
            for _, scores in self.attn_scores_1[layer_idx].items():
                all_scores_1 += scores[:self.ctxt_len]

            all_scores_2 = torch.zeros(self.ctxt_len)
            for _, scores in self.attn_scores_2[layer_idx].items():
                all_scores_2 += scores[:self.ctxt_len]

            if self.config.filter_special_tokens:
                mask = torch.ones(self.ctxt_len, dtype=torch.bool)
                for idx in range(self.ctxt_len):
                    if is_special_or_whitespace(idx, input_ids):
                        mask[idx] = False
                
                all_scores_1 = all_scores_1.masked_fill(~mask, float('-inf'))
                all_scores_2 = all_scores_2.masked_fill(~mask, float('-inf'))

            _, top_k_indices_1 = torch.topk(all_scores_1, k=self.config.top_k)
            _, top_k_indices_2 = torch.topk(all_scores_2, k=self.config.top_k)

            top_k_set_1 = set(top_k_indices_1.tolist())
            top_k_set_2 = set(top_k_indices_2.tolist())

            return top_k_set_1.intersection(top_k_set_2)

        content = f"{self.config.prefix_prompt_1}\n\n{context}"
        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids[0]

        for layer_idx in self.attn_scores_1.keys():
            sel = select_tokens(layer_idx, input_ids)
            self.sel_tokens[layer_idx] = sel
            
    def contextual_rereading(self, context: str):
        def hook(module: torch.nn.Module, input, output, layer_idx, is_first_pass, total_len):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    num_heads = attn_weights.shape[1]
                    for head_idx in range(num_heads):
                        attn_ctxt = attn_weights[0, head_idx, :total_len, :total_len]
                        sum_scores = attn_ctxt.sum(dim=0)
                        if is_first_pass:
                            self.attn_scores_1[layer_idx][head_idx] = sum_scores.cpu()
                        else:
                            self.attn_scores_2[layer_idx][head_idx] = sum_scores.cpu()

        def collect_attn_scores(context: str, is_first_pass: bool, cache_kv: bool = False):
            prefix_prompt = self.config.prefix_prompt_1 if is_first_pass else self.config.prefix_prompt_2
            content = f"{prefix_prompt}\n\n{context}"
            messages = [{"role": "user", "content": content}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            total_len = inputs.input_ids.shape[1]
            self.ctxt_len = total_len if is_first_pass else min(total_len, self.ctxt_len)

            registered_hooks = []
            for layer_idx, layer in enumerate(self.model.model.layers):
                hook_func = partial(
                    hook,
                    layer_idx=layer_idx,
                    is_first_pass=is_first_pass,
                    total_len=total_len
                )
                registered_hook = layer.self_attn.register_forward_hook(hook_func)
                registered_hooks.append(registered_hook)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    use_cache=True
                )

            for registered_hook in registered_hooks:
                registered_hook.remove()

            if cache_kv and is_first_pass:
                return text, outputs.past_key_values
            return None, None
        
        self.cached_text, self.cached_kv = collect_attn_scores(context, is_first_pass=True, cache_kv=True)
        collect_attn_scores(context, is_first_pass=False, cache_kv=False)
        self.compute_steering_matrices(context)
        self.original_cached_kv = copy.deepcopy(self.cached_kv)

    def generate_pred(self, question: str, context: List[str], dataset_name: str):
        context_text = ""
        if dataset_name == "squad" or dataset_name == "nq":
            context_text = context[0]
        elif dataset_name == "hotpotqa":
            context_text = "\n\n".join(context)
        if self.last_context is None or self.last_context != context_text:
            self.contextual_rereading(context_text)
            self.last_context = context_text

        answer = self._generate(question, max_new_tokens=50, temeprature=0.0)

    def generate_all_preds(self, dataset, dataset_name: str):
        samples = []
        predictions = []

        context_groups = defaultdict(list)
        for example in dataset:
            context_key = "\n\n".join(example["context"])
            context_groups[context_key].append(example)

        print(f"Processing {len(context_groups)} unique contexts...")

        for context_text, examples in tqdm(context_groups.items(), desc=f"Contexts"):
            print(f"Preparing context ({len(examples)} questions on this context)")
            self.contextual_rereading(context_text)
            for example in examples:
                answer = self._generate(example["question"], max_new_tokens=50, temperature=0.0)
                print(f"\nQ: {example['question']}")
                print(f"A: {answer}")

                predictions.append({
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"],
                    "prediction": answer,
                    "gold": example["answers"],
                })
        return predictions, samples


    def _generate(self, question: str, max_new_tokens: int = 50, **generate_kwargs):
        def hook(module: torch.nn.Module, input, output, layer_idx):

            if not isinstance(output, tuple) or len(output) < 2:
                return output

            attn_output = output[0]
            attn_weights = output[1]

            if attn_weights is None:
                return output

            if layer_idx not in self.sel_tokens:
                return output

            sel_token_indices = self.sel_tokens[layer_idx]
            if len(sel_token_indices) == 0:
                return output
           
            batch_size, num_heads, seq_len, total_len = attn_weights.shape

            M = torch.ones(seq_len, total_len, device=attn_weights.device)
            for token_idx in sel_token_indices:
                if token_idx < total_len:
                    M[:, token_idx] = self.config.alpha

            M = M.unsqueeze(0).unsqueeze(0)
            attn_weights_steered = attn_weights * M
            attn_weights_steered = attn_weights_steered / attn_weights_steered.sum(dim=-1, keepdim=True)
            return (attn_output, attn_weights_steered) + output[2:]
            
        question_content = f"{question}\n\nAnswer:"
        question_text = f"{self.cached_text}{question_content}"

        full_tokens = self.tokenizer(question_text, return_tensors="pt").to(self.device)

        cached_tokens = self.tokenizer(self.cached_text, return_tensors="pt").to(self.device)
        cache_len = cached_tokens.input_ids.shape[1]

        question_input_ids = full_tokens.input_ids[:, cache_len:]


        registered_hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook_func = partial(
                hook,
                layer_idx=layer_idx
            )
            registered_hook = layer.self_attn.register_forward_hook(hook_func)
            registered_hooks.append(registered_hook)

        past_kv = copy.deepcopy(self.original_cached_kv)
        try:
            current_input_ids = question_input_ids
            generated_ids = []
            temperature = generate_kwargs.get('temperature', 1.0)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True
                )

                logits = outputs.logits[:, -1, :]
                past_kv = outputs.past_key_values

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated_ids.append(next_token.item())
                current_input_ids = next_token

                for _ in range(1, max_new_tokens):
                    outputs = self.model(
                        input_ids=current_input_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=True,
                        return_dict=True
                    )
                    logits = outputs.logits[:, -1, :]
                    past_kv = outputs.past_key_values

                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)

                    generated_ids.append(next_token.item())

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    current_input_ids = next_token

            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()
        return generated_text.strip()


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random Seed for dataset sampling"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["squad", "hotpotqa", "nq"],
        help="Dataset to use",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k tokens to steer")
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    config = LLMSteerConfig(alpha=args.alpha, top_k=args.top_k, filter_special_tokens=False)
    llmsteer = LLMSteer(config, model_name)

    if args.dataset == "squad":
        dataset = QADatasetLoader.load_squad(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "hotpotqa":
        dataset = QADatasetLoader.load_hotpotqa(
            n_samples=args.n_samples, seed=args.seed
        )
    elif args.dataset == "nq":
        dataset = QADatasetLoader.load_natural_questions_mrqa(
            n_samples=args.n_samples, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    predictions, samples = llmsteer.generate_all_preds(dataset, args.dataset)

    config = {
        "dataset": args.dataset,
        "method": "llmsteer",
        "seed": args.seed,
        "n": args.n_samples,
        "alpha": args.alpha,
        "top_k": args.top_k
    }

    _, scores = QAEvaluator.evaluate(
        predictions, True, config 
    )
    print(scores)

if __name__ == "__main__":
    main()
