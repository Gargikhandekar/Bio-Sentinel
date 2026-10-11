"""Modern tabular model.

TabNet is used when PyTorch is available. On CPU-only laptops the fit is
restricted to a stratified subsample. If PyTorch is unavailable, a multilayer
perceptron with the same task head is used and the substitution is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier


@dataclass
class TabularFitResult:
    model: object
    name: str
    backend: str
    notes: str


def fit_modern_tabular(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_rows: int = 25000,
    max_epochs: int = 25,
) -> TabularFitResult:
    rng = np.random.default_rng(seed)
    if len(x_train) > max_rows:
        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]
        n_pos = min(len(pos), max_rows // 3)
        n_neg = min(len(neg), max_rows - n_pos)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(idx)
        x_fit, y_fit = x_train[idx], y_train[idx]
        note = f"Trained on stratified subsample n={len(idx)} because of the laptop CPU profile."
    else:
        x_fit, y_fit = x_train, y_train
        note = "Trained on the full training partition."

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch

        model = TabNetClassifier(
            seed=seed,
            verbose=0,
            device_name="cpu",
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.3,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 0.02},
        )
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_val, y_val)],
            max_epochs=max_epochs,
            patience=8,
            batch_size=2048,
            virtual_batch_size=256,
        )
        return TabularFitResult(model, "tabnet", "pytorch-tabnet", note)
    except Exception as exc:
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=40,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(x_fit, y_fit)
        return TabularFitResult(
            mlp,
            "mlp_tabular",
            "sklearn-mlp",
            note + f" TabNet unavailable ({type(exc).__name__}: {exc}). Substituted a CPU MLP.",
        )
