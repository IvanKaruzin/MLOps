FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-hw11.lock /app/requirements-hw11.lock
RUN pip install --no-cache-dir --require-hashes -r requirements-hw11.lock \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
COPY service.py /app/service.py
COPY hw11 /app/hw11
USER 10001:10001
EXPOSE 8000
CMD ["python", "-B", "service.py", "--backend", "simulated", "--host", "0.0.0.0", "--port", "8000"]
