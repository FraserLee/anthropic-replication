import datasets
import os

def get_dataset():

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cache/dataset_cache'))
    print(f"dataset_path: {dataset_path}")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # datasets.logging.set_verbosity(datasets.logging.ERROR)
    datasets.logging.set_verbosity(datasets.logging.INFO)

    train = datasets.load_dataset("c4", "en", streaming = True, split = "train")
    val   = datasets.load_dataset("c4", "en", streaming = True, split = "validation")

    # get just the text part of the dataset
    train = map(lambda x: x['text'], train)
    val   = map(lambda x: x['text'], val)

    print(next(iter(train)))

    return train, val

