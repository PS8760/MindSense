# MindSense — Streamlit app container.
#
# Lean runtime: installs only requirements.txt (no torch/transformers/training
# tooling). Raw and processed datasets are intentionally excluded (see
# .dockerignore); pages degrade gracefully without them.
#
# Container port defaults to 7860 (Hugging Face Spaces convention). For other
# targets either override at build time (`--build-arg PORT=8501`) or at run
# time (`-e PORT=8501`), then map the container port to the host port you want.
FROM python:3.11-slim

ARG PORT=7860
ENV PORT=${PORT} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Non-root runtime user (prompt: deployment must run as non-root).
RUN groupadd --system app && useradd --system --gid app app

# Python sources first: `-e .` in requirements.txt installs the package itself,
# so src/ must already be present. This layer changes rarely -> good caching.
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# App, config, models (mood-cue ONNX), reports and the native theme.
COPY .streamlit ./.streamlit
COPY app ./app
COPY config ./config
COPY reports ./reports
COPY --chown=app:app models ./models

RUN chown -R app:app /app
USER app

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; d=os.environ.get('PORT','7860'); r=urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health'%d,timeout=3); print('healthy', r.status); raise SystemExit(0 if r.status==200 else 1)"

CMD ["sh", "-c", "streamlit run app/Home.py --server.port=\"${PORT}\" --server.address=0.0.0.0"]