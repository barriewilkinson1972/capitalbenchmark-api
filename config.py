import os
from pathlib import Path


def benchmark_data_dir() -> Path:
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
            "context_visibility_pilot/gpt4o_mini"
        ),
        Path(
            "/Users/barrie/capitalbenchmark-api/"
            "benchmark_runs/context_visibility_pilot/gpt4o_mini"
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
    configured = os.getenv("BENCHMARK_HTML_DIR")

    if configured:
        path = Path(configured).expanduser().resolve()

        if not (path / "index.html").is_file():
            raise RuntimeError(
                "Configured BENCHMARK_HTML_DIR does not contain "
                f"index.html: {path}"
            )

        return path

    candidates = [
        # Local layout
        data_dir / "html"
    ]

    for path in candidates:
        path = path.resolve()

        if (path / "index.html").is_file():
            return path

    checked = ", ".join(
        str(path / "index.html")
        for path in candidates
    )

    raise RuntimeError(
        f"Benchmark frontend index.html was not found. Checked: {checked}"
    )