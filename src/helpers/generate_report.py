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
