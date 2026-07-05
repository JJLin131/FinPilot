FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY FinanceAgent ./FinanceAgent
COPY observability ./observability
COPY src/main/resources ./src/main/resources

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[eval]" \
    && mkdir -p /app/data /app/evals/datasets

EXPOSE 8099

CMD ["python", "-m", "FinanceAgent.main"]
