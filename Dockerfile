FROM python:3.11-slim

RUN pip install pixi

WORKDIR /app

COPY pixi.toml pixi.lock ./
COPY pyproject.toml ./
COPY src ./src

RUN pixi install
