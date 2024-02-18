import datasets
import tiktoken
import os

enc = tiktoken.get_encoding("cl100k_base")

def get_dataset():

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cache/dataset_cache'))
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    datasets.logging.set_verbosity(datasets.logging.ERROR)
    # datasets.logging.set_verbosity(datasets.logging.INFO)

    train = datasets.load_dataset("c4", "en", streaming = True, trust_remote_code = True, split = "train")
    val   = datasets.load_dataset("c4", "en", streaming = True, trust_remote_code = True, split = "validation")

    # filter to only text column
    train = map(lambda x: x['text'], train)
    val   = map(lambda x: x['text'], val)

    # tokenize
    train = map(lambda x: enc.encode(x), train)
    val   = map(lambda x: enc.encode(x), val)

    x = next(iter(train))
    print(x)
    print(enc.decode(x))

    return train, val

