FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y unoconv


WORKDIR /app

COPY ./canary_model/app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001
ENV PORT 5001

# Use python to execute the script instead of calling uvicorn directly
CMD exec uvicorn app:app --host 0.0.0.0 --port 5001