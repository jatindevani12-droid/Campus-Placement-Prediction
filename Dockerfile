FROM python:3.11-slim
WORKDIR /app
COPY . /app
# Install build tools temporarily to allow building wheels for packages that need C extensions,
# then purge them to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && \
	pip install --no-cache-dir -r requirements.txt && \
	apt-get purge -y --auto-remove build-essential gcc && rm -rf /var/lib/apt/lists/*
EXPOSE 8000
ENV PORT=8000
CMD ["python", "app.py"]
