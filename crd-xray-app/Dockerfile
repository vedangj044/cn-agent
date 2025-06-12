FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libpq-dev \
    sqlite3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./crd_xray_agent/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY crd_xray_agent/ crd_xray_agent/

COPY ./langgraph.json ./langgraph.json

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8080", "--no-reload"]