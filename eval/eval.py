import re
import string
from collections import Counter


class QAMetrics:
    """
    Calculate QA metrics using official evaluation methods
    Code for calculating metrics comes from SQuAD and HotpotQA eval scripts
    SQuAD: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    HotpotQA: https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
    """

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer string"""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return QAMetrics.normalize_answer(s).split()

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(
            QAMetrics.normalize_answer(a_gold) == QAMetrics.normalize_answer(a_pred)
        )

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = QAMetrics.get_tokens(a_gold)
        pred_toks = QAMetrics.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


class QAEvaluator:
    """Evaluate QA models on different datasets"""

    @staticmethod
    def evaluate(predictions):
        results = []
        exact_scores = []
        f1_scores = []
        for pred in predictions:
            # print("Prediction: ", pred)
            qid = pred["id"]
            gold_answers = [a for a in pred["gold"] if QAMetrics.normalize_answer(a)]
            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]
            # print("Gold: ", gold_answers, "\n")
            a_pred = pred["prediction"]
            # Take max over all gold answers
            exact_score = max(QAMetrics.compute_exact(a, a_pred) for a in gold_answers)
            f1_score = max(QAMetrics.compute_f1(a, a_pred) for a in gold_answers)

            results.append(
                {
                    "id": qid,
                    "question": pred["question"],
                    "prediction": pred["prediction"],
                    "gold": pred["gold"],
                    "exact_score": exact_score,
                    "f1_score": f1_score,
                }
            )
            exact_scores.append(exact_score)
            f1_scores.append(f1_score)

        aggr_exact_score = sum(exact_scores) / len(exact_scores) * 100
        aggr_f1_score = sum(f1_scores) / len(f1_scores) * 100
        scores = {"aggr_exact_score": aggr_exact_score, "aggr_f1_score": aggr_f1_score}
        return results, scores
