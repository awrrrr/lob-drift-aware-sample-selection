"""V1 classifiers for step 2, and the builders reused when the harness retrains.

All candidates share the same features and splits and use balanced class weights.
Every model exposes fit / predict / predict_proba with a consistent class order
(0 = Up, 1 = Stationary, 2 = Down), so the replay harness can retrain any of them
through the same interface.

The RBF SVM is wrapped so that it always fits on a fixed, seeded, stratified
subsample (config models.svm_subsample) and calibrates probabilities with
training-only CV. This keeps it affordable to retrain thousands of times in step 3.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

CLASSES = np.array([0, 1, 2])  # Up, Stationary, Down


def feature_columns(cfg):
    """Feature column names, with the step-1 constant features removed."""
    drop = set(cfg["models"]["drop_features"])
    return [f"f{i}" for i in range(cfg["data"]["n_features"]) if f"f{i}" not in drop]


def split_xy(df, days, cfg):
    """Rows for `days` that have a target. Returns (X, y, sample_id), time-ordered."""
    cols = feature_columns(cfg)
    sub = df[df.day.isin(days) & (df.target >= 0)].sort_values("sample_id")
    return sub[cols].values, sub.target.values, sub.sample_id.values


class SubsampledSVM(BaseEstimator, ClassifierMixin):
    """RBF SVM on a fixed seeded stratified subsample, with calibrated probabilities.

    Fitting on more than `subsample` rows draws exactly `subsample` of them
    (stratified, fixed seed) before fitting; fewer rows are all used. The fit is
    wrapped in CalibratedClassifierCV so predict_proba is a real probability, not
    a decision score. Calibration uses only the (subsampled) training data.
    """

    def __init__(self, C=1.0, gamma="scale", subsample=20000, calibration_cv=3,
                 calibration_method="sigmoid", seed=0):
        self.C = C
        self.gamma = gamma
        self.subsample = subsample
        self.calibration_cv = calibration_cv
        self.calibration_method = calibration_method
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if len(X) > self.subsample:
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=self.subsample, random_state=self.seed)
            idx, _ = next(splitter.split(X, y))
            X, y = X[idx], y[idx]
        base = SVC(C=self.C, gamma=self.gamma, kernel="rbf",
                   class_weight="balanced", random_state=self.seed)
        self.model_ = CalibratedClassifierCV(
            base, method=self.calibration_method, cv=self.calibration_cv)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_model(name, params, cfg):
    """Construct one candidate with the given grid params. Class order is CLASSES."""
    seed = cfg["models"]["seed"]
    if name == "logreg":
        # Multinomial is the default in this sklearn version (and the only option
        # going forward), so we do not pass multi_class explicitly.
        return LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=1000,
            n_jobs=-1, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"], weights=params["weights"], n_jobs=-1)
    if name == "svm_rbf":
        m = cfg["models"]
        return SubsampledSVM(
            C=params["C"], gamma=params["gamma"], subsample=m["svm_subsample"],
            calibration_cv=m["svm_calibration_cv"],
            calibration_method=m["svm_calibration_method"], seed=seed)
    raise ValueError(f"unknown model {name}")


def iter_grid(name, cfg):
    """Yield every param dict in a model's grid (cartesian product)."""
    grid = cfg["models"]["grids"][name]
    keys = list(grid)
    def rec(i, acc):
        if i == len(keys):
            yield dict(acc)
            return
        for v in grid[keys[i]]:
            acc[keys[i]] = v
            yield from rec(i + 1, acc)
    yield from rec(0, {})


def make_v1(cfg):
    """Build the frozen V1 (model + params) recorded in config models.v1."""
    v1 = cfg["models"]["v1"]
    if v1 is None:
        raise ValueError("models.v1 is not set; run scripts/train_v1.py first")
    return build_model(v1["model"], v1["params"], cfg)
