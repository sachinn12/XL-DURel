# XL-DURel

This repository contains the [**XL-DURel**](https://huggingface.co/sachinn1/xl-durel) along with instructions to reproduce the results presented in the paper: [XL-DURel: Finetuning Sentence Transformers for Ordinal Word-in-Context Classification](https://arxiv.org/pdf/2507.14578).

## Steps to Reproduce

Follow these steps to set up the environment and run the XL-DURel model:

### 2. Create a Virtual Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source ./venv/bin/activate
```
### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Obtain the Dataset

Please [contact us](https://www.ims.uni-stuttgart.de/institut/team/Schlechtweg/) to request access to the dataset.


### 4. Load the Dataset
After obtaining the dataset, provide the file paths in your notebook or script:

```bash
dev_df = pd.read_pickle("/path-to/dataset/dev.pkl")
test_df = pd.read_pickle("/path-to/dataset/test.pkl")
```

### 5. Run the Notebook

```bash
xl-durel.ipynb
```

## XL-DURel-2

[XL-DURel-2](https://huggingface.co/sachinn1/xl-durel2) is an extended version of the XL-DURel model. It has been trained on both the Ordinal/Binary WiC training and test splits.

## Results
To reproduce the plots reported in the paper, navigate to the results folder and open and run the notebook:
```bash
plot.ipynb
```
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

