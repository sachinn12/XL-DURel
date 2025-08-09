# XL-DURel

This repository contains the XL-DURel model and instructions to reproduce the results.

## Steps to Reproduce

Follow these steps to set up the environment and run the XL-DURel model:

### 1. Create a Virtual Environment

Create and activate a Python virtual environment to isolate dependencies:

```bash
python3 -m venv venv
source ./venv/bin/activate 
```

### 2. Install Required Package

```bash
pip install -r requirements.txt
```

### 3. Obtain the Dataset

Please [contact us](https://www.ims.uni-stuttgart.de/institut/team/Schlechtweg/) to request access to the dataset.

### 4. Once you have obtained the data, please provide the path to the dataset in the notebook.

Example:
```python
dev_df = pd.read_pickle("/projekte/cik/shared/llm/thesis/Semantic_Proximity/data/comedi-wic-mclwic/dev.pkl")
test_df = pd.read_pickle("/projekte/cik/shared/llm/thesis/Semantic_Proximity/data/comedi-wic-mclwic/test.pkl")
```
### 5. Run the notebook
