FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY irisen_model ./irisen_model
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

RUN python -m unittest

CMD ["python", "-m", "irisen_model"]

