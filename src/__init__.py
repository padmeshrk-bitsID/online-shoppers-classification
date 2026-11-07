"""Online Shoppers Purchasing Intention: binary classification package.

Module map (each module has exactly one responsibility):

    config.py         - paths, column groups and experiment settings
    data.py           - load raw CSV, validate schema, clean dtypes, split
    preprocessing.py  - build the feature-transformation step
    models.py         - one specification class per classifier (+ registry)
    training.py       - hyper-parameter tuning with cross-validation
    evaluation.py     - the six evaluation metrics + confusion matrix
    persistence.py    - save / load trained models and metadata
    workflow.py       - orchestrates the full training run
    inference.py      - prediction & evaluation service used by the app
    visualization.py  - reusable matplotlib figures
"""
