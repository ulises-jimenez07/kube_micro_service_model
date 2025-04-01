FROM python:3.9-slim

WORKDIR /app

COPY ./main_model/app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--proxy-headers"]