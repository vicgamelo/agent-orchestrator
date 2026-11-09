FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --all-extras --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --all-extras

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
COPY --from=build --chown=app:app /app /app
COPY --chown=app:app agents.yaml /app/agents.yaml
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PORT=8000 ORCH_CONFIG=/app/agents.yaml
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["agent-orchestrator"]
