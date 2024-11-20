from datasets import load_dataset


def get_data(path="openwebtext", split="train"):
    dataset = load_dataset(path=path, split=split).shuffle(seed=42)
    print(dataset)
    return dataset
