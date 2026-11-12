FROM astral/uv:python3.15-slim
WORKDIR /app
COPY . .
RUN uv sync
CMD ["uv", "run", "ollama_test.py"]