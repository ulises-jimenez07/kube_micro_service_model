FROM python:3.9-slim

WORKDIR /app

COPY ./elector/app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002", "--proxy-headers"]