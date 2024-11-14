# Classic methods

An end-to-end example:

```py
# data from: https://archive.ics.uci.edu/dataset/222/bank+marketing
# bank marketing
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

data = pd.read_csv(
    Path(
        r"bank+marketing\bank-additional\bank-additional\bank-additional.csv"
    ),
    sep=";",
)

# allows for an induction of the target
data = data.drop(columns=["duration"])

data = data.replace({"unknown": None})
print(f"{data.isna().sum().sum()} entries are missing")

# imbalanced
print(data.y.value_counts())

y = data.pop("y")
y = LabelEncoder().fit_transform(y)

# nominal features
nominal = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


# fill missing values with mode
features = SimpleImputer(strategy="most_frequent").fit_transform(data)
features = pd.DataFrame(features, columns=data.columns)

preprocessor = ColumnTransformer(
    [
        # one hot encode nominal features
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal),
        # scale euribor3m
        ("numeric", StandardScaler(), ["euribor3m"]),
        # bin age
        ("age", KBinsDiscretizer(n_bins=7), ["age"]),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("variance_threshold", VarianceThreshold()),
        ("classifier", None),
    ]
)
grid = [
    {
        "classifier": [
            RandomForestClassifier(random_state=42, class_weight="balanced")
        ]
    },
    {"classifier": [SVC(random_state=42, class_weight="balanced")]},
    {"classifier": [LogisticRegression(class_weight="balanced")]},
    {"classifier": [GradientBoostingClassifier(random_state=42)]},
]

search = GridSearchCV(
    pipe,
    grid,
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=654),
    scoring=["f1", "balanced_accuracy", "roc_auc"],
    refit="balanced_accuracy",
    verbose=2,
)

search.fit(features, y)
results = pd.DataFrame(search.cv_results_)
print(search.best_params_)
print(search.best_score_)
```