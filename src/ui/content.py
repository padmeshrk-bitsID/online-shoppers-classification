"""All user-facing wording in one place (no logic), so text can change without touching code."""

APP_TITLE = "Online Shopper Purchase Prediction"
APP_TAGLINE = (
    "Evaluate five pre-trained classifiers on your test data and see "
    "how well each one predicts whether a browsing session ends in a purchase."
)
HERO_CHIPS = ["UCI dataset #468", "Binary target: Revenue", "5 pre-trained models", "No retraining: inference only"]

# (icon, title, text) for the 3-step "how it works" strip
HOW_TO_STEPS = [
    ("📤", "Upload test data",
     "Use the Setup panel at the top. Until you upload a CSV, results are shown on the bundled test_data.csv."),
    ("🧠", "Pick a model",
     "Choose one of the five saved classifiers from the drop-down next to the upload box."),
    ("📊", "Explore the results",
     "Scroll down for the six metrics, confusion matrix, classification report, model comparison and predictions."),
]

UPLOAD_HELP = (
    "Upload the TEST split only (the free Streamlit tier has limited capacity). "
    "Required: all 17 input features + the 'Revenue' target column. Extra columns are ignored."
)
MODEL_HELP = "All five models were trained and tuned offline. The app only loads them and predicts; nothing is retrained."

MODEL_BLURBS = {
    "logistic_regression": "Linear model that turns a weighted sum of the features into a purchase probability.",
    "decision_tree": "Learns a tree of simple if-then rules; the easiest model to explain.",
    "knn": "Labels a session by looking at the most similar sessions in the training data.",
    "naive_bayes": "Probabilistic model that assumes features are independent and normally distributed.",
    "svm": "Finds the widest-margin boundary between buyers and non-buyers (RBF kernel).",
}

# metric -> (plain-English meaning, formula); shown as tooltips and in the glossary
METRIC_HELP = {
    "Accuracy": ("Share of all sessions classified correctly.", "(TP + TN) / all rows"),
    "AUC": (("How well the model ranks buyers above non-buyers across all thresholds "
             "(50% = random guessing, 100% = perfect)."),
            "Area under the ROC curve"),
    "Precision": ("Of the sessions predicted as buyers, the share that really bought.", "TP / (TP + FP)"),
    "Recall": ("Of the real buyers, the share the model caught.", "TP / (TP + FN)"),
    "F1": ("Balance between precision and recall (their harmonic mean).", "2 x Precision x Recall / (Precision + Recall)"),
    "MCC": (("Agreement between predictions and truth, from -100% (always wrong) through 0% (random) "
             "to +100% (perfect). Stays fair when classes are imbalanced."),
            "(TP x TN - FP x FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))"),
}

# column -> (type, accepted values / example); the CSV format guide
COLUMN_GUIDE = {
    "Administrative": ("integer >= 0", "3"),
    "Administrative_Duration": ("number >= 0 (seconds)", "45.5"),
    "Informational": ("integer >= 0", "0"),
    "Informational_Duration": ("number >= 0 (seconds)", "0"),
    "ProductRelated": ("integer >= 0", "27"),
    "ProductRelated_Duration": ("number >= 0 (seconds)", "1046.2"),
    "BounceRates": ("number 0 - 0.2", "0.004"),
    "ExitRates": ("number 0 - 0.2", "0.021"),
    "PageValues": ("number >= 0", "12.8"),
    "SpecialDay": ("number 0 - 1", "0"),
    "Month": ("text", "Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec"),
    "OperatingSystems": ("integer code", "1 - 8"),
    "Browser": ("integer code", "1 - 13"),
    "Region": ("integer code", "1 - 9"),
    "TrafficType": ("integer code", "1 - 20"),
    "VisitorType": ("text", "Returning_Visitor, New_Visitor, Other"),
    "Weekend": ("boolean", "TRUE / FALSE (also 1/0, yes/no)"),
    "Revenue": ("boolean target", "TRUE / FALSE (also 1/0, yes/no)"),
}

FAQ = [
    ("Why do I see results before uploading anything?",
     ("The app opens on the bundled held-out test split (test_data.csv) so you can explore straight away. "
      "Upload your own CSV and every result on the page switches to your file.")),
    ("Is any model retrained on my file?",
     ("No. The app loads the saved, already-trained pipelines from the `model/` folder and only predicts. "
      "Your file is never used for training.")),
    ("Which columns does my CSV need?",
     ("All 17 input features plus the target column `Revenue`. Column order doesn't matter and extra columns "
      "are ignored. See the format guide below.")),
    ("Why isn't accuracy the best metric here?",
     ("Only about 15 % of sessions end in a purchase, so a model that always says *no purchase* already scores "
      "about 85 % accuracy. F1 and MCC are fairer for imbalanced data.")),
    ("Why does AUC show 'n/a'?",
     "AUC needs both classes in the file. If every row has the same `Revenue` value, AUC is undefined."),
    ("What does the purchase score mean?",
     ("It is the model's predicted probability of a purchase (for the SVM it is the decision value). "
      "Higher means more likely to buy.")),
    ("Is my file stored?",
     "The app does not save uploaded files; they are processed in memory for your session only."),
]
