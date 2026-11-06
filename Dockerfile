# syntax=docker/dockerfile:1

# Build stage: install serving dependencies into a virtual environment
FROM python:3.12-slim-bookworm AS build
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serve.txt

# Runtime stage: copy the venv and application code only
FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib
# libgomp1 is required at runtime by scikit-learn/scipy OpenMP kernels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY . .
# Run as an unprivileged user that cannot modify the application files. The
# default 568 is the "apps" user on TrueNAS SCALE. It must be able to read the
# assets mounted from ASSETS_DIR.
ARG APP_UID=568
ARG APP_GID=568
RUN chmod -R a+rX /app \
    && groupadd --gid "$APP_GID" app \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --no-create-home --shell /usr/sbin/nologin app
USER ${APP_UID}:${APP_GID}
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
