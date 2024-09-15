FROM python:3.12-slim AS builder

RUN apt-get update && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
        curl \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_VERSION=1.8.3 \
    VENV_PATH="/usr/.venv"

RUN python3 -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install -U pip setuptools \
    && $VENV_PATH/bin/pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock /app/
RUN $VENV_PATH/bin/poetry install --no-dev --no-interaction --no-ansi


FROM python:3.12-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY ./ ./

RUN adduser --system --no-create-home user && chown -R user /app
USER user

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
