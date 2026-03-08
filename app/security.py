import os

from fastapi import Header, HTTPException, Query, status


def require_api_key(
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(default=None, alias="apiKey"),
) -> None:
    expected = os.getenv("EXPLAIN_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="EXPLAIN_API_KEY is not configured",
        )
    provided = x_api_key or api_key
    if not provided or provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
