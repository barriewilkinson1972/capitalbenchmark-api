import os
from pathlib import Path


def _contains_benchmark_frontend(path: Path) -> bool:
    """
    Return True when path contains either:
    - a legacy single frontend; or
    - one or more model-specific benchmark frontends.
    """
    if (path / "index.html").is_file():
        return True

    if (path / "html" / "index.html").is_file():
        return True

    return any(
        child.is_dir() and (child / "html" / "index.html").is_file()
        for child in path.iterdir()
    )


def benchmark_data_dir() -> Path:
    """
    Locate the benchmark root.

    Preferred layout:

        context_visibility_pilot/
            gpt4o_mini/
                html/
            gpt5/
                html/
    """
    configured = os.getenv("BENCHMARK_DATA_DIR")

    if configured:
        path = Path(configured).expanduser().resolve()

        if not path.is_dir():
            raise RuntimeError(
                f"Configured BENCHMARK_DATA_DIR does not exist: {path}"
            )

        return path

    candidates = [
        Path(
            "/opt/capitalbenchmark-data/"
            "context_visibility_pilot"
        ),
        Path(
            "/Users/barrie/capitalbenchmark-api/"
            "benchmark_runs/context_visibility_pilot"
        ),
    ]

    for path in candidates:
        path = path.expanduser().resolve()

        if path.is_dir():
            return path

    raise RuntimeError(
        "Could not locate the benchmark data directory. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def benchmark_html_dir(data_dir: Path) -> Path:
    """
    Return the root from which model-specific HTML directories are discovered.

    BENCHMARK_HTML_DIR may point to:
    - the benchmark root containing model folders;
    - a single model directory containing html/; or
    - a legacy html directory containing index.html.
    """
    configured = os.getenv("BENCHMARK_HTML_DIR")

    if configured:
        path = Path(configured).expanduser().resolve()

        if not path.is_dir() or not _contains_benchmark_frontend(path):
            raise RuntimeError(
                "Configured BENCHMARK_HTML_DIR does not contain a "
                f"benchmark frontend: {path}"
            )

        return path

    path = data_dir.expanduser().resolve()

    if _contains_benchmark_frontend(path):
        return path

    checked = [
        path / "index.html",
        path / "html" / "index.html",
        path / "<model>" / "html" / "index.html",
    ]

    raise RuntimeError(
        "Benchmark frontend index.html was not found. Checked: "
        + ", ".join(str(item) for item in checked)
    )
