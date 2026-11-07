# Online Shopper Purchase Prediction: Binary Classification

## a. Problem statement

Only about **1 in 6** browsing sessions on an e-commerce website ends in a purchase. If the site can tell
**during a session** whether a visitor is likely to buy, it can act on that: offer a discount or live chat to
undecided visitors, retarget likely buyers, or stop spending ad budget on sessions that won't convert.

**Objective:** using a session's browsing behaviour (pages visited and time spent), Google-Analytics page metrics
(bounce rate, exit rate, page value) and visit context (month, visitor type, traffic source etc.), build and compare
five classification models that predict whether the session **ends in a purchase (`Revenue = 1`) or not (`Revenue = 0`)**.
This is a **binary classification** problem.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository, dataset id 468: <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| Citation | Sakar, C. & Kastro, Y. (2018). UCI ML Repository. DOI: 10.24432/C5F88Q · License CC BY 4.0 |
| Instances | **12,330** sessions (minimum required: 500). 125 exact duplicates removed → 12,205 used |
| Features | **17** input features (minimum required: 12): 10 numeric + 7 categorical |
| Target | `Revenue` (True/False): did the session end in a purchase? |
| Class balance | 10,422 no purchase (84.5 %) / 1,908 purchase (15.5 %): **imbalanced** |
| Missing values | None |

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `Administrative` | Numeric | No. of administrative pages visited (account, login) |
| 2 | `Administrative_Duration` | Numeric | Time (sec) spent on administrative pages |
| 3 | `Informational` | Numeric | No. of informational pages visited (about, contact, policies) |
| 4 | `Informational_Duration` | Numeric | Time (sec) spent on informational pages |
| 5 | `ProductRelated` | Numeric | No. of product pages visited |
| 6 | `ProductRelated_Duration` | Numeric | Time (sec) spent on product pages |
| 7 | `BounceRates` | Numeric | Avg. bounce rate of the pages visited (Google Analytics) |
| 8 | `ExitRates` | Numeric | Avg. exit rate of the pages visited |
| 9 | `PageValues` | Numeric | Avg. value of the pages visited before a transaction |
| 10 | `SpecialDay` | Numeric | Closeness of the visit to a special day (0–1), e.g. Valentine's Day |
| 11 | `Month` | Categorical | Month of the visit |
| 12 | `OperatingSystems` | Categorical | Operating system (coded 1–8) |
| 13 | `Browser` | Categorical | Browser (coded 1–13) |
| 14 | `Region` | Categorical | Region of the visitor (coded 1–9) |
| 15 | `TrafficType` | Categorical | Traffic source (coded 1–20) |
| 16 | `VisitorType` | Categorical | Returning / New / Other visitor |
| 17 | `Weekend` | Boolean | Whether the visit was on a weekend |
| Target | `Revenue` | Boolean | Whether the session ended in a purchase |

**Pre-processing and training setup**
* Duplicate rows removed, then a **stratified 80/20 train-test split** (`random_state = 42`): 9,764 train / 2,441 test rows.
* Numeric features: `log1p` transform (heavy right skew) + `StandardScaler`.
* Categorical features: `OneHotEncoder` (rare levels grouped as "infrequent"); coded columns such as `Browser` are treated as categories, not numbers.
* Hyper-parameters tuned with `GridSearchCV` (5-fold stratified CV on the training set only), optimising **F1** because of the class imbalance.
* Preprocessing is inside each model's scikit-learn `Pipeline`, so it is fitted only on training data and saved with each model.

## c. GitHub Repository Link

**GitHub repository:** `<add your GitHub repository URL here>`

**Live Streamlit app:** `<add your Streamlit Community Cloud URL here>`

The repository contains all required files: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`,
the `model/` folder with all five trained models (`.joblib`), the source code (`src/`, `train_models.py`), and the
executed notebook as a PDF report (`ML_Assignment_Online_Shoppers_Classification.pdf`) that anyone can open and read.

## d. Models used

All five models were trained on the same dataset and evaluated on the same held-out test set (2,441 sessions, 382 purchases).

**Comparison table of evaluation metrics**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | 0.9205 | 0.5880 | 0.7958 | 0.6763 | 0.6157 |
| Decision Tree | 0.9021 | 0.9262 | 0.7227 | 0.6073 | 0.6600 | 0.6065 |
| KNN | 0.8939 | 0.9154 | 0.7043 | 0.5550 | 0.6208 | 0.5655 |
| Naive Bayes | 0.8578 | 0.8734 | 0.5347 | 0.7068 | 0.6088 | 0.5316 |
| Support Vector Machine | 0.8820 | 0.9255 | 0.5914 | 0.7958 | 0.6786 | 0.6183 |

Best per metric: Accuracy, AUC and Precision: Decision Tree · Recall: Logistic Regression & Support Vector Machine (tie) · F1 and MCC: Support Vector Machine.

Tuned hyper-parameters: Logistic Regression `C=0.003, class_weight='balanced'` · Decision Tree `max_depth=6, min_samples_leaf=60, criterion='gini'` ·
KNN `n_neighbors=35, weights='distance', p=2` · Naive Bayes (Gaussian) `var_smoothing=1.0` · Support Vector Machine (RBF kernel) `C=10, gamma=0.001, class_weight='balanced'`.

**Observations on the performance of each model**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Catches about **80 % of buyers** (Recall 0.7958, 304 of 382) at Precision 0.5880. F1 (0.6763) and MCC (0.6157) are almost the same as the SVM's, which shows that once the features are log-scaled the class boundary is close to linear. Strong regularisation (`C = 0.003`) with balanced class weights worked best. Tunes in ~4 s and its coefficients are interpretable, which makes it the best value for its cost. |
| Decision Tree | **Highest Accuracy (0.9021), AUC (0.9262) and Precision (0.7227)**: when it predicts "purchase" it is right 72 % of the time. It was not class-weighted, so it leans towards the majority class: Recall is only 0.6073 (misses 150 buyers). The shallow tree (depth 6, at least 60 samples per leaf) does not over-fit, and its first split is on `PageValues` (importance 0.78), which gives simple, explainable rules. |
| KNN | Good Precision (0.7043) but the **lowest Recall (0.5550)**, missing 170 of 382 buyers. In this 84/16 imbalanced data most neighbours of a buyer are non-buyers, and KNN has no class-weight option. It needs a large k (35, distance-weighted) to be stable, keeps all training rows in memory, and is slow at prediction time. |
| Naive Bayes | **Weakest overall**: lowest Accuracy (0.8578), AUC (0.8734) and MCC (0.5316). Both of its assumptions are violated here: features are strongly correlated (`BounceRates`–`ExitRates` r = 0.91, `ProductRelated`–`ProductRelated_Duration` r = 0.86), and one-hot features are not Gaussian. It needed heavy smoothing (`var_smoothing = 1.0`). Recall is reasonable (0.7068) and it is the fastest model to train. |
| Support Vector Machine | **Best F1 (0.6786) and MCC (0.6183)**, joint-best Recall (0.7958), and AUC 0.9255, almost equal to the Decision Tree's 0.9262. Balanced class weights plus a small `gamma` (0.001) with `C = 10` give a smooth, nearly linear boundary, which is why its results are close to Logistic Regression's. It is the most expensive model to tune (~3 min vs ~4 s for Logistic Regression). |
| Overall, Winner for your dataset? | **Support Vector Machine.** With 84.5 % non-buyers, accuracy is misleading: always predicting "no purchase" already scores 0.845. MCC and F1 are the fairest metrics here, and the SVM is best on both, catches ~80 % of buyers, and has a near-top AUC. **Logistic Regression** is a very close runner-up at a fraction of the cost. The Decision Tree is the better choice only if precision matters most (e.g. targeting costly discounts only at near-certain buyers). |

---

## Appendix: project details

### Streamlit app

The app loads the five **saved** models from `model/` and evaluates them on the test file. **No model is retrained on
the uploaded data.** An uploaded CSV must contain all **17 input features and the `Revenue` target column**.

**Layout: one page, nothing hidden.** There is no slide-out sidebar and no tabs. A fixed **Setup panel** at the top holds
the upload box and the model drop-down side by side. Below it, every result is shown straight away on the bundled
`test_data.csv`; uploading a CSV switches the whole page to your file. Jump links under the panel scroll to each section.

| Required feature | Where it is in the app |
|---|---|
| a. Dataset upload option (CSV, test data only) | **Setup panel, 1 · Test data**: drag-and-drop uploader, a note saying which data is shown, and a *Download sample test_data.csv* button |
| b. Model selection drop-down | **Setup panel, 2 · Model**: the five models, the best one marked ⭐; a card shows the model's description, tuned hyper-parameters, CV scores and the file it was loaded from |
| c. Display of evaluation metrics | **📊 Model performance** section: six animated cards (Accuracy, AUC, Precision, Recall, F1, MCC) shown as **percentages** (e.g. Accuracy 88.20 %), colour-coded Good / Moderate / Low, with hover explanations and a "best" tag; plus a chart of where the selected model stands among all five on each metric |
| d. Confusion matrix / classification report | **📊 Model performance** section: both, side by side, with a plain-English summary ("Caught 304 of 382 buyers…") |
| Results of all models on the test data | **🏆 Compare all models** section: winner per metric, overall winner (by MCC), an interactive ranking chart (pick any metric) next to a heatmap of all metrics, the full table with metric bars, and a CSV download |

**Ease of use**
* **Results by default**: the page opens on the bundled held-out test split, so the statistics are visible without any clicks. A badge row shows which file is being scored, how many sessions it has and how many are buyers.
* **Percentages everywhere in the app**: every metric (cards, tables, heatmap, CV scores) is shown as a percentage, e.g. 0.8820 becomes 88.20 %. MCC runs from -100 % to +100 % (0 % = no better than guessing). The comparison table in section d keeps the decimal format the assignment template asks for.
* **Colour-coded, interactive graphics** (hover any mark for details). Every colour also comes with a symbol and a text label, so colour is never the only cue:
  * **Performance bands**: ▲ Good (80 % and above, green), ● Moderate (60–80 %, amber), ▼ Low (below 60 %, red) on the metric cards and ranking bars, with a colour key.
  * **Where the model stands**: for each metric, the selected model's score (coloured dot) next to the other four models (grey dots) and their range, on a 0–100 % scale.
  * **Rank the models by a metric**: choose a metric and the five models re-order as colour-coded bars; the winner gets a 🏆 and the selected model is outlined.
  * **Where the predictions landed**: a stacked bar splitting every test session into correct no-purchase, correct purchase, false alarm and missed buyer, with counts and percentages.
* **Upload checks before scoring**: missing features, a missing `Revenue` column, empty cells, unreadable or non-numeric data each produce a clear message plus the expected format. Extra columns are ignored and reported.
* **Help where it's needed**: tooltips on every input, a *Metric guide* pop-over next to the metrics, and a **📘 Help** section at the bottom (how-to steps, metric glossary with formulas, CSV column guide, FAQ) in expandable boxes, the only collapsed part of the page.
* **🔍 Predictions** section: row-level results with quick filters (correct / misclassified / missed buyers / false alarms) and a CSV download.
* **Animations**: animated gradient header, cards that fade in one after another, metric bars that grow to their value, confusion-matrix cells that pop in, a step-by-step status box ("file checked → model loaded → metrics computed") and a toast when results are ready. All of them are switched off for users whose system asks for reduced motion.

### Project structure

```
project-folder/
├── app.py                    # Streamlit UI (presentation only)
├── train_models.py           # trains, evaluates and saves all five models
├── requirements.txt
├── README.md
├── test_data.csv             # held-out 20 % test split (2,441 rows), upload this in the app
├── .streamlit/config.toml    # app theme and upload limit
├── model/                    # saved trained models (each includes its fitted preprocessing)
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metadata.json         # input schema, tuned parameters, scores, library versions
│   └── model_comparison.csv  # test-set metrics of all models
├── src/                      # source code: one responsibility per module
│   ├── config.py  data.py  preprocessing.py  models.py  training.py
│   ├── evaluation.py  persistence.py  workflow.py  inference.py  visualization.py
│   └── ui/                   # app look & wording (no Streamlit imports, unit-tested)
│       ├── theme.py          #   stylesheet: colours, cards, animations
│       ├── components.py     #   HTML blocks: header, colour-coded metric cards, charts, confusion matrix
│       └── content.py        #   help text, metric glossary, CSV guide, FAQ
├── ML_Assignment_Online_Shoppers_Classification.pdf   # executed notebook as a PDF report (code + outputs + charts)
├── data/online_shoppers_intention.csv                             # raw UCI dataset
└── tests/                    # pytest suite (data, models, metrics, saved models, UI blocks)
```

### How to run

```bash
pip install -r requirements.txt
streamlit run app.py          # start the app (uses the saved models)
python train_models.py        # optional: retrain all models (~5 min)
pip install pytest && pytest  # optional: run the tests
```

Python 3.11–3.13. The saved models were created with scikit-learn 1.9.1 (pinned in `requirements.txt`).

**Deploy:** push the folder to GitHub (including the hidden `.streamlit/` folder) → <https://share.streamlit.io> →
*Create app* → select the repo and `app.py` → Advanced settings: Python 3.12 → *Deploy*.

### Design (SOLID principles)

| Principle | Where it shows |
|---|---|
| Single responsibility | Each `src/` module has one job (cleaning, preprocessing, tuning, evaluation, persistence); `src/ui/` builds the look and wording; `app.py` only arranges the page. |
| Open/closed | Add a model with one `ModelSpec` subclass in `MODEL_SPECS`, or a metric with one `Metric` entry (plus its help text), without changing other code. |
| Liskov substitution | Every `ModelSpec` is interchangeable; every saved model is a `Pipeline` with the same `predict` interface. |
| Interface segregation | A spec exposes only `build_estimator()` and `param_grid()`; the app talks only to `PredictionService`. |
| Dependency inversion | `TrainingWorkflow` and `PredictionService` receive their collaborators through the constructor, so each part can be swapped or tested on its own. |
