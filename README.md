# Negative frequency-dependent selection in post-vaccine pneumococcal population

This repository contains resources to replicate the parameter inference experiments reported by Corander et al. [1]. 
The inference procedure is implemented with ELFI [2] in the attached notebook.

The notebook is tested with Python 3.5 and ELFI v. 0.7.4.

## Files and folders

- **code/simulator/**
  This folder contains the multilocus NFDS simulator.
- **code/utils.py**
  Contais operations which are used in parameter inference with ELFI and in visualisation.
- **data/**
  This folder contains the observed data.
- **inference.ipynb**
  Jupyter notebook defining the ELFI model and running the inference.

## References

- [1] Corander et al. (2017). Frequency-dependent selection in vaccine-associated pneumococcal population dynamics. Nature Ecology and Evolution. 1(12):1950–1960. doi:10.1038/s41559-017-0337-x
- [2] Lintusaari et al. (2018). ELFI: Engine for Likelihood Free Inference.
  Journal of Machine Learning Research. 19(16):1-7. Code https://github.com/elfi-dev/elfi.
