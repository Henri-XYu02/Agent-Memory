FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Default environment variables (can be overridden)
ENV USER_ID=default \
    FLASK_PORT=5001 \
    AGENT_PROVIDER=bedrock \
    AGENT_MODEL=claude-3-sonnet

EXPOSE 5001
CMD ["python", "-m", "app.api.agent_service"]


