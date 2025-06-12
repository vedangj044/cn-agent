FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libpq-dev \
    sqlite3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY crd-xray/ crd-xray/

CMD ["kopf", "run", "crd-xray/server.py", "--standalone"]