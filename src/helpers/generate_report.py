import subprocess
from datetime import datetime
from pathlib import Path


def generate_report(notebook_path: str, output_path: str = "reports") -> None:
    """
    Function to render a Jupyter notebook as a static HTML report and save
    it to the specified path.
    """
    notebook_file = Path(notebook_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = notebook_file.name[:-6]  # Remove .ipynb extension
    html_filename = f"{name}_{timestamp}.html"
    html_path = output_dir / html_filename
    html_path = html_path.absolute()

    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            # "--execute", # TODO: Fix execution issues later
            "--to",
            "html",
            "--output",
            str(html_path),
            str(notebook_file),
        ],
        check=True,
    )

    _add_timestamp_footer(html_path, timestamp)


def _add_timestamp_footer(html_path: Path, timestamp: str) -> None:
    """Add a timestamp footer to the HTML report."""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    footer = f"\n<hr><p><small>Report generated: {formatted_time}</small></p>"

    content = content.replace("</body>", f"{footer}\n</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def report_model_evaluation(experiment):
    "Print a comprehensive report of the model evaluation results for a given experiment."
    experiment.show_train_test_genrefreq_comparison()
    print("Random Baseline Evaluation:")
    print("-" * 40)
    experiment.show_random_baseline_evaluation()

    print("\n\n")
    print("Tuning History:")
    print("-" * 40)
    experiment.show_tuning_history()

    print("\n\n")
    print("Model evaluation on Holdout Set:")
    print("-" * 40)
    experiment.show_model_evaluation()

    print("\n\n")
    print("Top Coefficients per Genre:")
    print("-" * 40)
    _show_top_coefficients_per_genre(experiment.model_coefficients, top_n=10)


def _show_top_coefficients_per_genre(coeffs, top_n=10):
    """Show the most important features for each genre based on the logistic regressions' model coefficients."""
    genres = coeffs.columns
    for genre in genres:
        print(f"Top {top_n} coefficients for genre: {genre.upper()}")
        top_coeffs = coeffs[genre].abs().sort_values(ascending=False).head(top_n)
        for feature, value in top_coeffs.items():
            print(f"{feature} ({coeffs.at[feature, genre]:.3f})")
        print("\n")
