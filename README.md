# CC-Diff: Enhancing Contextual Coherence in Remote Sensing Image Synthesis


## Installation

### Conda environment setup
```
conda create -n CC-Diff python=3.9 -y
conda activate CC-Diff
pip install -r requirement.txt
```

## Inference

```
python infer_dior.py
```

## Data preparation

This is an example:
```
DIOR
├── train
│   ├── 00003.jpg
|   ├── ...
|   ├── metadata.jsonl
├── val
|   ├── 00011.jpg
|   ├── ...
|   ├── metadata.jsonl
├── results
│   ├── ...
├── dior_emb.pt
```
Here are more details.

## Training
```
./dist_train.sh
```

## Acknowledgements