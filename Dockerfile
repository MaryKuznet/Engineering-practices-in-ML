FROM python:3.11-slim

RUN pip install pixi

WORKDIR /app

COPY pixi.toml pixi.lock pyproject.toml ./
COPY . .
#COPY src ./src

RUN pixi install

CMD ["bash"]
