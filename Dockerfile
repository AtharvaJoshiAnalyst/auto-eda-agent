FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY core ./core
COPY agent ./agent
COPY .streamlit ./.streamlit
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]