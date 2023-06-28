# A model-agnostic heuristics for Selective Classification


Data can be found [here](https://www.dropbox.com/sh/bvhrjdjkj1dyzry/AACsMi2IBKFIWqPoFlDJRhyHa?dl=0).


## System specifics

All the code was run on a machine with Ubuntu 20.04.4 and using programming language Python 3.8.12.

## Usage

Download this repository from github and then place downladed data in 'code/data'.
We suggest to create a new environment using:

```bash
 $ conda create -n ENVNAME --file environment.yml
  ```
Activate environment and go to the code folder by using:

```bash
 $ conda activate ENVNAME
 $ cd code
  ```


To run experiments on tabular data for Table 1

- For PlugIn and SCross
  ```bash
  $ python exp_realdata.py --model lgbm --boot_iter 1000 --cv 5
  ```


- For SAT

  ```bash
  $ python exp_realdata.py --model resnet --metas sat --boot_iter 1000 --max_epochs 300
  ```
  
- For SELNET

  ```bash
  $ python exp_realdata_selnet.py --model resnet --boot_iter 1000 --max_epochs 300
  ```
To run experiments for CatsVsDogs for Table 1 (check the paths):

- for SCROSS:
   ```bash
  $ python exp_catsdogs_scross.py
  ```
- for PLUGIN
  ```bash
  $ python exp_catsdogs_plugin.py
   ```
- for SAT and SELNET:
  ```bash
  $ python exp_catsdogs_selnet.py
  ```


To run experiments on tabular data for Table 2:
  ```bash
  $ python exp_realdata.py --model lgbm --boot_iter 1000 --cv DESIRED_K
  ```
To run experiments on CatsVsDogs for Table 2:
  ```bash
  $ python exp_catsdogs_scross.py --boot_iter 1000 --cv DESIRED_K
  ```


To run experiments for Table 3, possible DESIRED_BASE_CLASSIFIER: xgboost, rf, resnet, logistic.
  ```bash
  $ python exp_realdata.py --model DESIRED_BASE_CLASSIFIER --boot_iter 1000 --cv 5 
  ```
