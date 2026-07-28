from pathlib import Path

from flask import abort, send_from_directory
from werkzeug.utils import safe_join


ALLOWED_FRONTEND_SUFFIXES = {
    ".html",
    ".css",
    ".js",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".map",
}


def register_frontend_routes(app):

    html_dir = Path(app.config["HTML_DIR"])


    def resolve_frontend_file(filename: str) -> Path:

        requested_filename = str(filename).strip()

        if not requested_filename:
            abort(
                400,
                description="A frontend filename is required.",
            )

        resolved_value = safe_join(
            str(html_dir),
            requested_filename,
        )

        if resolved_value is None:
            abort(
                400,
                description="Invalid frontend file path.",
            )

        resolved_path = Path(resolved_value)

        if resolved_path.suffix.lower() not in ALLOWED_FRONTEND_SUFFIXES:
            abort(
                403,
                description="Unsupported frontend file type.",
            )

        if not resolved_path.is_file():
            abort(
                404,
                description=f"Frontend file not found: {requested_filename}",
            )

        return resolved_path


    @app.get("/")
    def frontend_home():

        index_path = html_dir / "index.html"

        if not index_path.is_file():
            abort(
                404,
                description="Benchmark frontend index.html was not found.",
            )

        response = send_from_directory(
            directory=str(html_dir),
            path="index.html",
            mimetype="text/html",
            conditional=True,
            max_age=300,
        )

        response.headers["X-Content-Type-Options"] = "nosniff"

        return response


    @app.get("/<path:filename>")
    def frontend_file(filename: str):

        requested_path = resolve_frontend_file(filename)

        relative_path = requested_path.relative_to(html_dir)

        response = send_from_directory(
            directory=str(html_dir),
            path=str(relative_path),
            conditional=True,
            max_age=3600,
        )

        response.headers["X-Content-Type-Options"] = "nosniff"

        return response