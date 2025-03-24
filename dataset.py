import json
import torch
from torch.utils.data import Dataset
import itertools
from copy import deepcopy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
# from multiprocessing import Pool, cpu_count


class CoconutDataset(Dataset):
    def __init__(
        self,
        name,
        train_or_val,
        modality,
        stage,
        debug,
        tokenizer,
        start_id,
        end_id,
        latent_id,
        c,
    ):
        self.stage = stage
        self.train_or_val = train_or_val
        self.modality = modality
        self.name = name
        self.debug = debug
        self.tokenizer = tokenizer
        self.start_id = start_id
        self.latent_id = latent_id
        self.end_id = end_id
        self.c = c
        self.raw_data = self.load_data(name, train_or_val, debug)
        self.tokenized_data = self.tokenize_data()
        self.data = self.process_tokens()

    def load_data(self, name, train_or_val, debug):
        with open(f"data/{name}_{train_or_val}.json") as f:
            data = json.load(f)
        if debug:
            data = data[:200] if train_or_val == "train" else data[:20]
        return data

    def tokenize_sample(self, idx, sample):
        question_tokenized = self.tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            self.tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = self.tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]

        tokenized_sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": idx,
        }
        return tokenized_sample

    def tokenize_data(self):
        # with Pool(16) as p:
        #     return list(tqdm(p.starmap(self.tokenize_sample, enumerate(self.raw_data)), total=len(self.raw_data)))
        return [self.tokenize_sample(idx, sample) for idx, sample in enumerate(self.raw_data)]

    def _get_test_tokens(self, tokenized_sample):
        tokens = tokenized_sample["question_tokenized"]
        self.n_latent = (
            min(self.stage, len(tokenized_sample["steps_tokenized"])) * self.c
        )
        if self.modality == "coconut":  # if coconut, add latents after question
            tokens += [self.start_id] + [self.latent_id] * self.n_latent + [self.end_id]
        return tokens

    def _get_train_tokens(self, tokenized_sample):
        tokens = []
        if self.modality == "cot":
            tokens += list(itertools.chain.from_iterable(tokenized_sample["steps_tokenized"])) # add all steps if cot
        elif self.modality == "coconut":
            tokens += list(itertools.chain.from_iterable(tokenized_sample["steps_tokenized"][
                min(self.stage, len(tokenized_sample["steps_tokenized"])) :
            ]))  # add only some steps if coconut, previous one are latents
        tokens += tokenized_sample["answer_tokenized"]
        return tokens

    def _get_labels_tokens(self, tokenized_sample, train_tokens):
        n_question_tokens = len(tokenized_sample["question_tokenized"])
        labels = [-100] * n_question_tokens
        if self.modality == "coconut":
            # skip start and end and latent tokens, and the question
            labels += [-100] * (2 + self.n_latent) 
            labels += train_tokens[n_question_tokens + self.n_latent + 2:] 
        else:
            # only skip the question (add steps + answer for cot and just answer for base)
            labels += train_tokens[n_question_tokens:]  
        return labels

    def process_tokens(self):
        data = []
        for idx, tokenized_sample in enumerate(self.tokenized_data):
            data.append(
                {
                    "idx": idx,
                    "input_ids": self._get_test_tokens(
                        deepcopy(tokenized_sample)
                    ),  # Get question tokens (or question + latent tokens if coconut)
                }
            )
        if self.train_or_val == "train":
            for s in data:
                s["input_ids"] += self._get_train_tokens(
                    self.tokenized_data[s["idx"]]
                )  # Add steps tokens (if cot or coconut) and answer tokens for training
                s["labels"] = self._get_labels_tokens(
                    self.tokenized_data[s["idx"]], s["input_ids"]
                )  # Add labels for training
        for s in data:
            s["attention_mask"] = [1] * len(s["input_ids"])
            s["position_ids"] = list(range(len(s["input_ids"])))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

# We need to define a custom collate_fn so that elements in a batch are of equal size.
# We first pad to the left so that the latent tokens are aligned. We then pad to the rigth.
def collate_fn(data, tokenizer, latent_id):
    earliest_latent = [
        s["input_ids"].index(latent_id) for s in data if latent_id in s["input_ids"]
    ]

    if len(earliest_latent) > 0:  # There are latent tokens
        for s in data:
            if latent_id in s["input_ids"]:
                n_pad = max(earliest_latent) - s["input_ids"].index(latent_id)
            else:
                n_pad = 0

            s["position_ids"] = [0] * n_pad + s["position_ids"]
            s["input_ids"] = [tokenizer.pad_token_id] * n_pad + s["input_ids"]
            if "labels" in s:
                s["labels"] = [-100] * n_pad + s["labels"]
            s["attention_mask"] = [0] * n_pad + s["attention_mask"]

    non_label_position_features = [
        {k: v for k, v in s.items() if k not in ["labels", "position_ids"]}
        for s in data
    ]

    # run through tokenizer without labels to ensure no side effects
    batch = pad_without_fast_tokenizer_warning(
        tokenizer,
        non_label_position_features,
        padding=True,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    if "labels" in data[0]:
        labels = [s["labels"] for s in data]
        max_label_length = max(len(l) for l in labels)
        batch["labels"] = [
            label + [-100] * (max_label_length - len(label)) for label in labels
        ]
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

    position_ids = [s["position_ids"] for s in data]
    max_pos_length = max(len(l) for l in position_ids)

    batch["position_ids"] = [
        position_id + [0] * (max_pos_length - len(position_id))
        for position_id in position_ids
    ]
    batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.int64)

    return batch