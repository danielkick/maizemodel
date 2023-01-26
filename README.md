MaizeModel
==============================

This repository is a copy of the one I created for [Kick et al. 2023.](https://academic.oup.com/g3journal/advance-article/doi/10.1093/g3journal/jkad006/6982634). For project reconstruction, I advise one download the files from Zenodo and follow the setup instructions. Due to the data's scale it is not included here. The code and setup instructions can be found here (https://zenodo.org/record/7401113) and the cleaned data is available here (https://zenodo.org/record/6916775). 

Project Organization
------------
    ├── README.md <- top-level overview
    │
    ├── data <------ Empty folder for generated figures and table. This folder 
    │                should be filled using data from zenodo 10.5281/zenodo.6916775
    │
    ├── models
    │   ├── 0_hp_search_syr_G    │ Hyperparamter search               │ Deep Neural
    │   ├── 0_hp_search_syr_S    │ G:Genomic, S:Soil,                 │ Network 
    │   ├── 0_hp_search_syr_W    │ W:Weather + Management,            │ related
    │   ├── 0_hp_search_syr_cat  │ cat:DNN-CO, full:DNN-SO. "syr" is  │ folders
    │   ├── 0_hp_search_syr_full │ site-year rebalanced.              │ 
    │   │                                                             │ 
    │   │                                                             │ 
    │   ├── 3_finalize_model_syr__rep_G    │ Model Finalization using │ 
    │   ├── 3_finalize_model_syr__rep_S    │ hyperparameters found in │ 
    │   ├── 3_finalize_model_syr__rep_W    │ 0_hp_search_syr_*        │ 
    │   ├── 3_finalize_model_syr__rep_cat  │                          │ 
    │   ├── 3_finalize_model_syr__rep_full │                          │
    │   │
    │   ├── 3_finalize_lm                  │ Simple fixed effects models (R)
    │   │
    │   ├── 3_finalize_model_BGLR_BLUPS    │ BLUPs, subdirs with different models (R)
    │   │
    │   ├── 3_finalize_classic_ml          │ Machine learning 
    │   ├── 3_finalize_classic_ml_10x      │ hyperparameter selection
    │   ├── 9_classic_ml                   │ and replicate training
    │   │
    │   └── alt_weather                    │ Demo weather matrix generation
    │
    ├── notebooks <------------------------ Data cleaning and setup, figure generation
    │
    └── output <--------------------------- Empty folder for generated figures and tables

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
