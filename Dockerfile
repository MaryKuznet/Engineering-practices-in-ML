FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Pixi install script expects bash
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app

# Для кеша: сначала только файлы окружения
COPY pixi.toml pixi.lock* pyproject.toml ./
RUN pixi install

# Код проекта
COPY . .
# Скрипты
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY run_container.sh /usr/local/bin/run_container.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/run_container.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/usr/local/bin/run_container.sh"]
