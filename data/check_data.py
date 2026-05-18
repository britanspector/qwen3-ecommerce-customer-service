from datasets import load_dataset

data_files = {
    "train": "data/ecommerce_sft_trl_messages_strict20_5800_train.jsonl",
    "eval": "data/ecommerce_sft_trl_messages_strict20_5800_eval.jsonl",
}

dataset = load_dataset("json", data_files=data_files)

print(dataset)
print(dataset["train"][0])
print(dataset["train"][0]["messages"])