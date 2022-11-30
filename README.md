# SCross


This is the repository for the paper [A model agnostic heuristics for Selective Classification]().

To run experiments on tabular data for Table 1:


- For PlugIn and SCross

> python exp_realdata.py --model lgbm --boot_iter 1000 --cv 5

- For SAT

> python exp_realdata.py --model resnet --metas sat --boot_iter 1000 --max_epochs 300

- For SELNET

> python exp_realdata_selnet.py --model resnet --boot_iter 1000 --max_epochs 300

To run experiments for CatsVsDogs for Table 1 (check the paths):

- for SCROSS:

> python exp_catsdogs_scross.py

- for PLUGIN

> python exp_catsdogs_plugin.py

- for SAT and SELNET:

> python exp_catsdogs_selnet.py



To run experiments on tabular data for Table 2:

> python exp_realdata.py --model lgbm --boot_iter 1000 --cv DESIRED_K

To run experiments on CatsVsDogs for Table 2:

> python exp_catsdogs_scross.py --boot_iter 1000 --cv DESIRED_K



To run experiments for Table 3, possible DESIRED_BASE_CLASSIFIER: xgboost, rf, resnet, logistic.

> python exp_realdata.py --model DESIRED_BASE_CLASSIFIER --boot_iter 1000 --cv 5 
