# aiml_propagation

This package contains scripts and routines for AI/ML-based propagation prediction.

## Installation
1. git clone repo: 
    a. ```git clone https://bitbucket.fcc.gov/scm/~jonathan.lu/aiml_propagation.git```
2. pip install package in editable mode
    1. change directory to top level (directory of pyproject.toml)
    2. pip install dependencies in requirements.txt
        1. normal: ```py -m pip install -r requirements.txt```
        2. ssl error: ```py -m pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user -r requirements.txt```
    3. pip install package
        1. normal: ```py -m pip install -e .```
        2. ssl error: ```py -m pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user -e .```

## Common Usage
git config http.postBuffer 524288000
## Tests

