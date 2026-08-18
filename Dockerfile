FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system fontprint && useradd --system --gid fontprint --create-home fontprint
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install '.[api]'

USER fontprint
EXPOSE 8000
ENTRYPOINT ["fontprint"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
