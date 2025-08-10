# Overview

This package provides utility functions to support running the **xl-durel** model.

It includes:

1. Tokenization with target word marking, truncation, and decoding with special tokens.  
2. Calculation of Spearman and Pearson correlation coefficients.  
3. Computation of Krippendorff’s alpha with optimized thresholding — based on [CoMeDi Shared Task](https://aclanthology.org/2025.comedi-1.4.pdf).  
4. Heatmap visualization of evaluation metrics.

All functions are bundled in a Python package available on PyPI and can be installed via:

```bash
pip install xl-durel-utils
