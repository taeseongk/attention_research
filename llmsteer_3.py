import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from dataclasses import dataclass
from collections import defaultdict
from functools import partial
import types
import copy
import argparse
from tqdm import tqdm

from data import QADatasetLoader
from eval import QAEvaluator
from util import apply_template 

@dataclass
class LLMSteerConfig:
    alpha: float
    top_k: int
    filter_special_tokens: bool = True
    prefix_prompt_1: str = """Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words.
Context:"""

    prefix_prompt_2: str = """Answer the question based on the given passages below. Only give me the answer and do not output any other words. The following are given passages.
Context:"""

#    prefix_prompt_2: str = """Respond to the question using the context provided. Give only the answer with no extra words.
#Context:"""

class LLMSteer:
    """
    LLMSteer implementation from the paper.
    """

    def __init__(self, config: LLMSteerConfig, head_config, model_name: str):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.device = "cuda"

        self.attn_scores_1 = defaultdict(lambda: defaultdict(list))
        self.attn_scores_2 = defaultdict(lambda: defaultdict(list))
        self.ctxt_len = None
        self.context_len = None
        self.sel_tokens = {}

        self.cached_kv = None
        self.cached_text = None
        self.original_cached_kv = None

        self.last_context = None

        self.original_forwards = {}
        self.is_patched = False

        self.cached_prompt = None

        self.head_config = head_config

        
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

        def select_tokens(layer_idx: int, context_start: int, context_end: int):
            context_len = context_end - context_start
            all_scores_1 = torch.zeros(context_len)
            for _, scores in self.attn_scores_1[layer_idx].items():
                all_scores_1 += scores[context_start:context_end]

            all_scores_2 = torch.zeros(context_len)
            for _, scores in self.attn_scores_2[layer_idx].items():
                all_scores_2 += scores[context_start:context_end]

            if self.config.filter_special_tokens:
                mask = torch.ones(context_len, dtype=torch.bool)
                for idx in range(context_len):
                    if is_special_or_whitespace(context_start+idx, input_ids):
                        mask[idx] = False
                
                all_scores_1 = all_scores_1.masked_fill(~mask, float('-inf'))
                all_scores_2 = all_scores_2.masked_fill(~mask, float('-inf'))

            _, top_k_indices_1 = torch.topk(all_scores_1, k=self.config.top_k)
            _, top_k_indices_2 = torch.topk(all_scores_2, k=self.config.top_k)

            top_k_set_1 = set(top_k_indices_1.tolist())
            top_k_set_2 = set(top_k_indices_2.tolist())
            intersection = top_k_set_1.intersection(top_k_set_2)
        
            intersection_absolute = {idx + context_start for idx in intersection}
            return intersection_absolute

        prompt = f"{self.config.prefix_prompt_1}\n\n{context}"
        text = apply_template(prompt, self.tokenizer, False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids[0]

        prefix_prompt = f"{self.config.prefix_prompt_1}\n\n"
        prefix_text = apply_template(prefix_prompt, self.tokenizer, False)
        prefix_tokens = self.tokenizer(prefix_text, return_tensors="pt").to(self.device)
        ctxt_start = prefix_tokens.input_ids.shape[1]
        ctxt_end = input_ids.shape[0]
        
        for layer_idx in self.attn_scores_1.keys():
            sel = select_tokens(layer_idx, ctxt_start, ctxt_end)
            #filtered_sel = {pos for pos in sel if pos >= prefix_len}
            #self.sel_tokens[layer_idx] = filtered_sel
            self.sel_tokens[layer_idx] = sel 

        # Debug: Print selected tokens
        print(f"\n=== Selected Tokens Summary ===")
        print(f"Context: {context}...")
        print(f"Context length: {len(context)} chars")
        print(f"Total tokens in sequence: {self.ctxt_len}")
        print(f"\nSelected tokens per layer:")
        #for layer_idx in sorted(self.sel_tokens.keys())[:3]:  # Show first 3 layers
        for layer_idx in sorted(self.sel_tokens.keys())[:]:  # Show first 3 layers
            tokens = self.sel_tokens[layer_idx]
            print(f"Layer {layer_idx}: {len(tokens)} tokens selected")
            if tokens:
                sample_positions = sorted(list(tokens))[:]
                sample_tokens = [self.tokenizer.decode([input_ids[i]]) for i in sample_positions]
                print(f"  Sample: {sample_tokens}")
            
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

        def collect_attn_scores(context: str, is_first_pass: bool):
            prefix_prompt = self.config.prefix_prompt_1 if is_first_pass else self.config.prefix_prompt_2
            prompt = f"{prefix_prompt}\n\n{context}"
            text = apply_template(prompt, self.tokenizer, False)

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            total_len = inputs.input_ids.shape[1]
            print("Hello1:", total_len)
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
                self.model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False
                )

            for registered_hook in registered_hooks:
                registered_hook.remove()


        self.context_len = self.tokenizer(context, return_tensors="pt").to(self.device).input_ids.shape[1]
        collect_attn_scores(context, is_first_pass=True)
        collect_attn_scores(context, is_first_pass=False)
        self.compute_steering_matrices(context)
        self._patch_attn_layers()

        prefix_prompt = self.config.prefix_prompt_1
        prompt = f"{prefix_prompt}{context}"
        self.cached_prompt = prompt
        self.cached_text = apply_template(prompt, self.tokenizer, False)
        inputs = self.tokenizer(self.cached_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                output_attentions=False
            )
        
        self.cached_kv = outputs.past_key_values
        self.original_cached_kv = copy.deepcopy(self.cached_kv)

        #self._patch_attn_layers()
        #self._unpatch_attn_layers()

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
        question = f"Question: {question}\n\nAnswer:"
        full_prompt = f"{self.cached_prompt}\n\n{question}"

        full_text =  apply_template(full_prompt, self.tokenizer, True)
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        cached_tokens = self.tokenizer(self.cached_text, return_tensors="pt").to(self.device)
        cache_len = cached_tokens.input_ids.shape[1]
        question_input_ids = full_tokens.input_ids[:, cache_len:]

        past_kv = copy.deepcopy(self.original_cached_kv)

        current_input_ids = question_input_ids
        generated_ids = []
        temperature = generate_kwargs.get('temperature', 1.0)

        with torch.no_grad():
            outputs = self.model(
                input_ids=current_input_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
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
                    output_attentions=False,
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
                
        return generated_text.strip()

    def _patch_attn_layers(self):
        def make_steered_attention_forward(layer_idx):
            def steered_eager_attention(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=0.0,
                **kwargs,
            ):
                key_states = repeat_kv(key, module.num_key_value_groups)
                value_states = repeat_kv(value, module.num_key_value_groups)

                attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                
                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask

                if layer_idx in self.sel_tokens:
                    sel_token_indices = self.sel_tokens[layer_idx]
                    
                    if len(sel_token_indices) > 0:
                        batch, num_heads, seq_len, total_len = attn_weights.shape
                        
                        if self.head_config and layer_idx in self.head_config:
                            heads_to_steer = self.head_config[layer_idx]
                        else:
                            heads_to_steer = list(range(num_heads))
                        
                        # Create M with BATCH dimension
                        M = torch.ones(
                            batch, num_heads, seq_len, total_len,
                            device=attn_weights.device,
                            dtype=attn_weights.dtype
                        )
                        
                        # Apply steering
                        for head_idx in heads_to_steer:
                            if head_idx >= num_heads:  # Safety check
                                continue
                            for token_idx in sel_token_indices:
                                if token_idx >= total_len:  # Safety check
                                    continue
                                M[:, head_idx, :, token_idx] = self.config.alpha
                        
                        attn_weights = attn_weights * M

                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
                attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()

                return attn_output, attn_weights 
            return steered_eager_attention
        
        def make_forward(steered_attention_fn):
            def forward(
                self_attn,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self_attn.head_dim)
                
                query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                
                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self_attn.layer_idx, cache_kwargs
                    )
                
                attn_output, attn_weights = steered_attention_fn(
                    self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
                    scaling=self_attn.scaling,
                    **kwargs,
                )
                
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self_attn.o_proj(attn_output)
                
                return attn_output, attn_weights
            
            return forward
        
        steering_layers = range(32) if self.head_config is None else self.head_config.keys()

        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx not in steering_layers:
                continue
            self.original_forwards[layer_idx] = layer.self_attn.forward
            steered_attention_fn = make_steered_attention_forward(layer_idx)
            layer.self_attn.forward = types.MethodType(
                make_forward(steered_attention_fn),
                layer.self_attn
            )
        self.is_patched = True

    def _unpatch_attn_layers(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in self.original_forwards:
                layer.self_attn.forward = self.original_forwards[layer_idx]
        
        self.is_patched = False
        self.original_forwards = {}
            
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
    head_config = {
            26: [0, 8, 16, 24],
            27: [4, 12, 20, 28],
            28: [2, 10, 18, 26],
            29: [6, 14, 22, 30],
            30: [1, 9, 17, 25],
            31: [3, 11, 19, 27],
        }
    config = LLMSteerConfig(alpha=args.alpha, top_k=args.top_k, filter_special_tokens=True)
    llmsteer = LLMSteer(config, head_config, model_name)

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
        "method": "llmsteer/3",
        "seed": args.seed,
        "n": args.n_samples,
        "alpha": args.alpha,
        "top_k": args.top_k
    }
    QAEvaluator.save_samples(samples, config)
    _, scores = QAEvaluator.evaluate(
        predictions, True, config 
    )
    print(scores)

if __name__ == "__main__":
    main()
