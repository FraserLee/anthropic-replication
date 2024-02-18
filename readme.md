Repo for replication of 
[*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html).

## Setup

```bash
pipenv install
```

##### Setup specific to Ubuntu Ubuntu 20.04 LTS

```bash
sudo apt-get install libcudnn8
sudo apt-get install libcudnn8-dev
```

## Usage

```bash
pipenv run python src/main.py
```

This will give you a local version of python, which you invoke by running, say, `python main.py` and where you can install packages by going `pip install`. 
