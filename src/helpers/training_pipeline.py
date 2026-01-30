"""Training pipeline for genre classification with Bayesian optimization."""

import pickle
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from .adaptive_sampling import AdaptiveSampler
from .bayesian_optimization import BayesianOptimizer


def create_pipeline_factory(
    random_state: int = 42,
) -> Callable[[dict[str, float]], ImbPipeline]:
    """Create pipeline factory for model training.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Factory function that creates pipeline from parameters
    """

    def factory(params: dict[str, float]) -> ImbPipeline:
        return ImbPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "sampler",
                    AdaptiveSampler(
                        target_ratio=params["target_ratio"],
                        random_state=random_state,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=params["C"],
                        l1_ratio=params["l1_ratio"],
                        penalty="elasticnet",
                        solver="saga",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    return factory


def train_model_with_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_space: dict[str, list[float]] | None = None,
    n_initial: int = 25,
    n_iterations: int = 50,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    checkpoint_dir: str | None = None,
) -> tuple[ImbPipeline, dict]:
    """Train model using Bayesian optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        param_space: Parameter search space (log scale for C and target_ratio)
        n_initial: Number of initial samples for Latin hypercube
        n_iterations: Number of Bayesian optimization iterations
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs for initial phase (-1 for all cores)
        checkpoint_dir: Directory for checkpoints (None to disable)

    Returns:
        Tuple of (best_pipeline, optimization_results)
    """
    if param_space is None:
        param_space = {
            "C": [-3, 2],
            "l1_ratio": [0, 1],
            "target_ratio": [np.log10(1.2), np.log10(3.0)],
        }

    pipeline_factory = create_pipeline_factory(random_state)

    optimizer = BayesianOptimizer(
        param_space=param_space,
        n_initial=n_initial,
        n_iterations=n_iterations,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=n_jobs,
        checkpoint_dir=checkpoint_dir,
    )

    results = optimizer.optimize(pipeline_factory, X_train, y_train)

    print("Selecting best parameters according to 1-SE rule...")
    best_pipeline = pipeline_factory(results["best_params"])
    print("Retraining best pipeline on full training data...")
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, results


def evaluate_model(
    pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Evaluate trained model on test data.

    Args:
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with predictions, metrics, and confusion matrix
    """
    y_pred = pipeline.predict(X_test)

    class_report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    conf_mat = confusion_matrix(y_test, y_pred)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "predictions": y_pred,
        "classification_report": class_report,
        "confusion_matrix": conf_mat,
        "macro_f1": macro_f1,
    }


def save_training_artifacts(
    pipeline: ImbPipeline,
    optimization_results: dict,
    evaluation_results: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "../../models/fell_sporleder_baseline",
) -> Path:
    """Save all training artifacts to disk.

    Args:
        pipeline: Trained pipeline
        optimization_results: Results from optimization
        evaluation_results: Results from evaluation
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Output directory path

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    coefficients = pd.DataFrame(
        {
            "feature": [
                f"feature_{i}"
                for i in range(pipeline.named_steps["classifier"].coef_.shape[1])
            ],
            "coefficient": pipeline.named_steps["classifier"].coef_[0],
        }
    )
    coefficients.to_csv(output_path / "coefficients.csv", index=False)

    train_split = pd.DataFrame({"label": y_train})
    train_split.to_csv(output_path / "train_split.csv", index=False)

    test_split = pd.DataFrame({"label": y_test})
    test_split.to_csv(output_path / "test_split.csv", index=False)

    with open(output_path / "optimization_results.pkl", "wb") as f:
        pickle.dump(optimization_results, f)

    with open(output_path / "evaluation_results.pkl", "wb") as f:
        pickle.dump(evaluation_results, f)

    return output_path


def visualize_results(
    evaluation_results: dict,
    y_test: pd.Series,
    output_dir: str | None = None,
) -> plt.Figure:
    """Generate and save visualization of model results.

    Args:
        evaluation_results: Results from evaluate_model
        y_test: Test labels
        output_dir: Optional directory to save plots

    Returns:
        Matplotlib figure object
    """
    conf_mat = evaluation_results["confusion_matrix"]
    labels = sorted(y_test.unique())

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if output_dir:
        plt.savefig(Path(output_dir) / "confusion_matrix.png", dpi=300)

    plt.show()

    class_report_df = pd.DataFrame(
        evaluation_results["classification_report"]
    ).transpose()

    fig, ax = plt.subplots(figsize=(10, 8))
    metrics_to_plot = class_report_df.loc[labels, ["precision", "recall", "f1-score"]]
    metrics_to_plot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics")
    ax.legend(title="Metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_dir:
        plt.savefig(Path(output_dir) / "per_class_metrics.png", dpi=300)

    plt.show()

    return fig
