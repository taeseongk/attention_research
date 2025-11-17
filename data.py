"""
Multi-Dataset QA Evaluation Script
"""

from datasets import load_dataset
from util import clean_context

import json
import gzip
import urllib.request
from pathlib import Path
import random


class QADatasetLoader:
    """Load and prepare different QA datasets"""

    @staticmethod
    def load_squad(split="validation", n_samples=None, seed=None):
        """
        Load SQuAD dataset
        SQuAD: Reading comprehension dataset with questions based on Wikipedia
        """
        print(f"Loading SQuAD ({split})...")
        dataset = load_dataset("squad", split=split)

        if n_samples and n_samples < len(dataset):
            if seed is not None:
                random.seed(seed)
                indices = random.sample(range(len(dataset)), n_samples)
                dataset = dataset.select(indices)
            else:
                dataset = dataset.select(range(n_samples))

        # Format for unified interface
        formatted_data = []
        for item in dataset:
            # Filter to answerables only
            formatted_data.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "context": [item["context"]],
                    "answers": (
                        item["answers"]["text"] if item["answers"]["text"] else []
                    ),
                    "dataset": "squad",
                }
            )

        return formatted_data

    @staticmethod
    def load_hotpotqa(split="validation", n_samples=None, seed=None):
        """
        Load HotpotQA dataset
        HotpotQA: Multi-hop question answering dataset
        """
        print(f"Loading HotpotQA ({split})...")
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split)

        if n_samples and n_samples < len(dataset):
            if seed is not None:
                random.seed(seed)
                indices = random.sample(range(len(dataset)), n_samples)
                dataset = dataset.select(indices)
            else:
                dataset = dataset.select(range(n_samples))

        # Format for unified interface
        formatted_data = []
        for item in dataset:
            # Concatenate all context paragraphs
            contexts = []
            for _, sentences in zip(
                item["context"]["title"], item["context"]["sentences"]
            ):
                context = "".join(sentences)
                contexts.append(context)

            formatted_data.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "context": contexts,
                    "answers": [item["answer"]],
                    "dataset": "hotpotqa",
                    "type": item["type"],  # 'bridge' or 'comparison'
                }
            )

        return formatted_data

    @staticmethod
    def load_natural_questions_mrqa(split="dev", n_samples=None, seed=None):
        """Load MRQA version of NQ"""
        print(f"Loading Natural Questions dataset (MRQA {split})...")

        urls = {
            'train': 'https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz',
            'dev': 'https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz'
        }

        url = urls[split]
        cache_dir = Path.home() / '.cache' / 'mrqa'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"NaturalQuestionsShort_{split}.jsonl.gz"

        # Download if not cached
        if not cache_file.exists():
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, cache_file)
            print("Download complete!")
        else:
            print("Using cached file...")

        # Read JSONL
        all_data = []
        with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
            # Skip header line
            header = json.loads(f.readline())

            for line in f:
                item = json.loads(line)
                for qa in item["qas"]:
                    all_data.append({
                        "id": qa["qid"],
                        "question": qa["question"],
                        "context": [clean_context(item["context"])],
                        "answers": [qa["detected_answers"][0]['text']] if qa["detected_answers"] else qa["answers"],
                        "dataset": "natural_questions"
                    })

        print(f"Total available: {len(all_data)} examples")

        # Random sampling with seed
        if seed:
            random.seed(seed)
            data = random.sample(all_data, min(n_samples, len(all_data)))
        else:
            data = all_data[:n_samples]

        print(f"Loaded {len(data)} examples")
        return data

        
def main():
    squad_dataset = QADatasetLoader.load_squad(n_samples=10)
    hotpot_dataset = QADatasetLoader.load_hotpotqa(n_samples=10)
    nq = QADatasetLoader.load_natural_questions_mrqa(n_samples=10)
    #print(squad_dataset[0])
    #print(hotpot_dataset[0])
    print(nq)


if __name__ == "__main__":
    main()
