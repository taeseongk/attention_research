"""
Multi-Dataset QA Evaluation Script
"""

from datasets import load_dataset
import random


class QADatasetLoader:
    """Load and prepare different QA datasets"""

    @staticmethod
    def load_squad(split="validation", n_samples=100, seed=None):
        """
        Load SQuAD v2 dataset
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
            # print(item['answers']['text'])
            formatted_data.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "context": item["context"],
                    "answers": (
                        item["answers"]["text"] if item["answers"]["text"] else []
                    ),
                    "dataset": "squad",
                }
            )

        return formatted_data

    @staticmethod
    def load_hotpotqa(split="validation", n_samples=100):
        """
        Load HotpotQA dataset
        HotpotQA: Multi-hop question answering dataset
        """
        print(f"Loading HotpotQA ({split})...")
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split)

        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        # Format for unified interface
        formatted_data = []
        for item in dataset:
            # Concatenate all context paragraphs
            context = " ".join(
                [
                    " ".join(sentences)
                    for title, sentences in zip(
                        item["context"]["title"], item["context"]["sentences"]
                    )
                ]
            )

            formatted_data.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "context": context,
                    "answers": [item["answer"]],
                    "dataset": "hotpotqa",
                    "type": item["type"],  # 'bridge' or 'comparison'
                }
            )

        return formatted_data


def main():
    squad_dataset = QADatasetLoader.load_squad(n_samples=10)
    hotpot_dataset = QADatasetLoader.load_hotpotqa(n_samples=10)
    print(squad_dataset[0])
    print(hotpot_dataset[0])


if __name__ == "__main__":
    main()
