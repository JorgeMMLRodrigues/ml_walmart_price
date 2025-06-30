import time, json, csv, inspect
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

# Add imports for halving search
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

class ModelTrainer:
    def __init__(
        self,
        X, y,
        model_factory,
        search: str = "grid",
        param_grid: dict = None,
        n_iter: int = 20,
        cv_splitter=None,
        custom_metrics: dict = None,
        imbalance_sampler=None,
        log_path: str = "results.csv",
        model_name: str = "Model",
        problem_type: str = "auto",
        compute_confusion: bool = False,
        n_jobs: int = 1,
        random_state: int = None
    ):
        """
        General-purpose model trainer with grid, random, and halving search,
        CV, imbalance sampling, custom metrics, threshold tuning, and CSV logging.
        """
        self.X = X
        self.y = y
        self.model_factory = model_factory
        self.search = search
        self.param_grid = param_grid or {}
        self.n_iter = n_iter
        self.cv_splitter = cv_splitter
        self.custom_metrics = custom_metrics or {}
        self.imbalance_sampler = imbalance_sampler
        self.log_path = log_path
        self.model_name = model_name
        self.problem_type = problem_type
        self.compute_confusion = compute_confusion
        self.n_jobs = n_jobs
        self.random_state = random_state

        # build combos
        if self.search == "grid":
            self.param_list = list(ParameterGrid(self.param_grid))
        elif self.search == "random":
            self.param_list = list(ParameterSampler(
                self.param_grid, self.n_iter, random_state=self.random_state
            ))
        elif self.search == "halving":
            # halving search will be performed in train()
            self.param_list = None
        else:
            raise ValueError("search must be 'grid', 'random', or 'halving'")

        # detect problem type if needed
        if self.problem_type == "auto":
            first = self.param_list[0] if self.param_list else {}
            try:
                mdl = self.model_factory(first).fit(self.X.iloc[:2], self.y.iloc[:2])
                self.problem_type = (
                    "clf" if getattr(mdl, "_estimator_type", "") == "classifier"
                    else "reg"
                )
            except:
                self.problem_type = "reg"

        # built-in metrics
        if self.problem_type == "reg":
            self.builtin_metrics = {
                "mae":  lambda yt, yp: mean_absolute_error(yt, yp),
                "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                "r2":   lambda yt, yp: r2_score(yt, yp)
            }
        else:
            self.builtin_metrics = {
                "accuracy": lambda yt, yp: accuracy_score(yt, yp),
                "f1":       lambda yt, yp: f1_score(yt, yp, average="macro"),
                "roc_auc":  lambda yt, yp: roc_auc_score(yt, yp)
            }

        # combine metrics
        self.metrics = {**self.builtin_metrics, **self.custom_metrics}

        # prepare CSV
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.log_path, "w", newline="")
        header = ["model_name", "param_json"] + list(self.metrics.keys())
        if self.problem_type == "clf" and self.compute_confusion:
            header += ["tn", "fp", "fn", "tp"]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=header)
        self.writer.writeheader()

    def _get_folds(self):
        n = len(self.X)
        # single hold-out if float
        if isinstance(self.cv_splitter, float):
            cut = int(n * (1 - self.cv_splitter))
            yield np.arange(cut), np.arange(cut, n)
        # sklearn splitter
        elif hasattr(self.cv_splitter, "split"):
            for tr, val in self.cv_splitter.split(self.X, self.y):
                yield tr, val
        # list of tuples
        elif isinstance(self.cv_splitter, (list, tuple)):
            for tr, val in self.cv_splitter:
                yield tr, val
        else:
            # default 80/20 holdout
            cut = int(n * 0.8)
            yield np.arange(cut), np.arange(cut, n)

    def train(self):
        # handle halving search separately
        if self.search == "halving":
            return self._train_halving()

        total = len(self.param_list)
        pbar = tqdm(self.param_list,
                    desc=f"{self.model_name} search",
                    unit="combo")
        best_score, best_params = float("inf"), None

        for params in pbar:
            t0 = time.time()
            scores = {m: [] for m in self.metrics}
            conf_sum = np.zeros(4, int)

            for tr_idx, val_idx in self._get_folds():
                X_tr, y_tr = self.X.iloc[tr_idx], self.y.iloc[tr_idx]
                X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]

                if self.imbalance_sampler is not None:
                    X_tr, y_tr = self.imbalance_sampler.fit_resample(X_tr, y_tr)

                model = self.model_factory(params)
                model.fit(X_tr, y_tr)

                if self.problem_type == "clf":
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X_val)[:, 1]
                        thr = getattr(model, "_decision_thr", 0.5)
                        y_pred = (prob >= thr).astype(int)
                    else:
                        y_pred = model.predict(X_val)
                        prob = None
                else:
                    y_pred, prob = model.predict(X_val), None

                # compute metrics
                for name, fn in self.metrics.items():
                    sig = inspect.signature(fn).parameters
                    try:
                        if len(sig) == 2:
                            m = fn(y_val, y_pred)
                        elif len(sig) == 3:
                            m = fn(y_val, y_pred, prob)
                        else:
                            m = np.nan
                    except:
                        m = np.nan
                    scores[name].append(m)

                # confusion if requested
                if self.problem_type == "clf" and self.compute_confusion:
                    tn, fp, fn_, tp = confusion_matrix(y_val, y_pred).ravel()
                    conf_sum += np.array([tn, fp, fn_, tp], int)

            # aggregate
            row = {"model_name": self.model_name,
                   "param_json": json.dumps(params)}
            for name in self.metrics:
                row[name] = float(np.mean(scores[name]))
            if self.problem_type == "clf" and self.compute_confusion:
                row.update({
                    "tn": int(conf_sum[0]),
                    "fp": int(conf_sum[1]),
                    "fn": int(conf_sum[2]),
                    "tp": int(conf_sum[3]),
                })

            self.writer.writerow(row)
            self.csv_file.flush()

            # update best
            prim = list(self.metrics)[0]
            if row[prim] < best_score:
                best_score, best_params = row[prim], params

            # ETA
            elapsed = time.time() - t0
            rem = elapsed * (total - (pbar.n))
            pbar.set_postfix({prim: f"{row[prim]:.4f}", "eta": f"{rem:.1f}s"})

        self.csv_file.close()
        return best_params, best_score

    def _train_halving(self):
            """
            A manual successive‐halving implementation that shows a tqdm bar + ETA
            at each budget level.
            """
            # 1) Prepare the initial random candidate list (exclude max_iter here)
            base_params = {k: v for k, v in self.param_grid.items() if k != "max_iter"}
            budgets     = sorted(self.param_grid["max_iter"])
            # Draw n_iter random combos of the OTHER params
            candidates = list(ParameterSampler(base_params, self.n_iter, random_state=self.random_state))

            # 2) Decide on the primary metric for elimination
            prim = list(self.metrics)[0]           # e.g. "wmae"
            # We assume lower-is-better for reg, except if prim=="r2"
            minimize = True if (self.problem_type=="reg" and prim!="r2") else False

            # 3) Run successive halving
            for budget in budgets:
                next_round = []
                pbar = tqdm(
                    candidates,
                    desc=f"{self.model_name} halving ({budget} iters)",
                    unit="combo"
                )
                for params in pbar:
                    # inject this budget
                    params_full = {**params, "max_iter": budget}
                    # evaluate over CV folds to get primary metric
                    scores = []
                    for tr_idx, val_idx in self._get_folds():
                        X_tr, y_tr = self.X.iloc[tr_idx], self.y.iloc[tr_idx]
                        X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]
                        model = self.model_factory(params_full)
                        model.fit(X_tr, y_tr)
                        y_pred = model.predict(X_val)
                        scores.append(self.metrics[prim](y_val, y_pred))
                    avg_score = float(np.mean(scores))
                    # update bar with current score
                    pbar.set_postfix({prim: f"{avg_score:.4f}"})
                    # collect for next round
                    next_round.append((avg_score, params))

                # 4) eliminate worst half
                # for minimize tasks (like mae, wmae), sort ascending; else descending
                next_round.sort(key=lambda x: x[0], reverse=(not minimize))
                keep_n = max(1, len(next_round)//2)
                candidates = [p for _, p in next_round[:keep_n]]

            # 5) after final budget, the top candidate is best
            best_params = {**candidates[0], "max_iter": budgets[-1]}
            best_score  = next_round[0][0]  # primary metric of top combo
            return best_params, best_score
