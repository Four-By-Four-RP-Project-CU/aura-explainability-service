from pathlib import Path

from fastapi import HTTPException, status
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent.parent


def safe_file_response(path: str):
    candidate = (BASE_DIR / path).resolve()
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(candidate)
