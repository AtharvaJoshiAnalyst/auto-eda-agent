FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY .streamlit ./.streamlit
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
CMD ["streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]
