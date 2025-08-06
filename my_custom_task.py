from lm_eval.base import Task, rf
from lm_eval.metrics import exact_match

# This is a base class that we can reuse to avoid repeating code.


class BaseTask(Task):
    VERSION = 0
    # The rest of the logic is the same for all tasks, so we define it here.

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, ["\n"])
        return completion

    def process_results(self, doc, results):
        completion = results[0]
        ground_truth = self.doc_to_target(doc)
        return {"exact_match": exact_match(completion, ground_truth)}

    def aggregation(self):
        return {"exact_match": "mean"}

    def higher_is_better(self):
        return {"exact_match": True}

# --- Define your three specific tasks below ---

# Task 1: high_to_high


class HighToHighTask(BaseTask):
    DATASET_PATH = "data/high_to_high.jsonl"

# Task 2: high_to_low


class HighToLowTask(BaseTask):
    DATASET_PATH = "data/high_to_low.jsonl"

# Task 3: high_to_medium


class HighToMediumTask(BaseTask):
    DATASET_PATH = "data/high_to_medium.jsonl"

# This function returns a dictionary of all tasks in this file.
# This is the method used by your version of the library to find tasks.


def create_all_tasks():
    return {
        "high_to_high": HighToHighTask,
        "high_to_low": HighToLowTask,
        "high_to_medium": HighToMediumTask,
    }
