# [XL-DURel](https://huggingface.co/sachinn1/xl-durel)

This repository contains the instructions to reproduce the results presented in the paper: [XL-DURel: Finetuning Sentence Transformers for Ordinal Word-in-Context Classification](https://arxiv.org/pdf/2507.14578).

## Steps to Reproduce

Follow these steps to set up the environment and run the XL-DURel model:
### 1. Clone this repo

```bash
git clone https://github.com/sachinn12/XL-DURel.git
```

### 2. Create a Virtual Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source ./venv/bin/activate
```
### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Obtain the Dataset

Please [contact us](https://www.ims.uni-stuttgart.de/institut/team/Schlechtweg/) to request access to the dataset.


### 5. Load the Dataset
After obtaining the dataset, provide the file paths in your notebook or script:

```bash
dev_df = pd.read_pickle("/path-to/dataset/dev.pkl")
test_df = pd.read_pickle("/path-to/dataset/test.pkl")
```

### 6. Run the Notebook

```bash
xl-durel.ipynb
```

## XL-DURel-2

[XL-DURel-2](https://huggingface.co/sachinn1/xl-durel2) is an extended version of the XL-DURel model. It has been trained on both the Ordinal/Binary WiC training and test splits.

### Citation

```bash
@misc{yadav2025xldurelfinetuningsentencetransformers,
      title={XL-DURel: Finetuning Sentence Transformers for Ordinal Word-in-Context Classification}, 
      author={Sachin Yadav and Dominik Schlechtweg},
      year={2025},
      eprint={2507.14578},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.14578}, 
}
```

