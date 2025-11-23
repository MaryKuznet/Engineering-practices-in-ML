# Отчёт по Домашнему Заданию №1  

---

# 📘 Содержание
1. [Структура проекта](#структура-проекта-2-балла)  
2. [Качество кода](#качество-кода-2-балла)  
3. [Управление зависимостями](#управление-зависимостями-2-балла)  
4. [Git workflow](#git-workflow-1-балл)
5. [Заключение](#заключение)

---

# Структура проекта 

## 1.1 Создание структуры через Cookiecutter

Для создания шаблонной структуры ML-проекта был использован `cookiecutter`.  
Для создания структуры проекта была использована команда
```
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```
И убрала ненужные на мой взгляд файлы

---

## 1.2 Настройка шаблонов для новых проектов

Создан отдельный шаблон в директории:

```
project_template/
├── cookiecutter.json
└── {{cookiecutter.project_name}}/
    ├── data/
    ├── notebooks/
    ├── src/
    ├── tests/
    └── README.md
```

Файл `cookiecutter.json`:

```json
{
  "project_name": "ml_project",
  "author_name": "Maria",
  "description": "A new ML project."
}
```

Создание нового проекта:

```
cookiecutter ./project_template
```

---

## 1.3 README

Создан README, описывающий:

- предметную область (Titanic — классификация выживания)
- цели проекта
- технологический стек
- структуру проекта

---

# Качество кода

## 2.1 Настройка pre-commit hooks

Создан файл `.pre-commit-config.yaml` со следующими инструментами:

- **Black** — автоформатирование  
- **isort** — сортировка импортов  
- **Ruff** — линтер  
- **MyPy** — статическая типизация  
- **Bandit** — анализ безопасности  

Установка:

```
pixi run pre-commit install
```

Запуск проверки:

```
pixi run pre-commit run --all-files
```

---

## 2.2 Форматирование кода (Black, isort, Ruff)

Настройки находятся в `pyproject.toml`:

```toml
[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 88
select = ["E", "F", "W"]
ignore = ["E501"]
```

---

## 2.3 Линтеры (Ruff, MyPy, Bandit)

## 2.3 Линтеры (Ruff, MyPy, Bandit)

### Ruff
Ruff используется как основной инструмент для линтинга Python-кода.  
Он проверяет стиль, ошибки, предупреждения и частично заменяет flake8, isort и другие инструменты.

Ruff запускается автоматически через pre-commit при каждом коммите.

Запуск вручную:

```bash
pixi run ruff check .
```

### MyPy
MyPy — инструмент статической типизации, который проверяет корректность аннотаций типов в проекте.

Запуск вручную:
```
pixi run mypy src
```

### Bandit:  
Bandit — линтер для поиска уязвимостей в Python-коде (например, небезопасного использования eval, subprocess и т.д.).

Он также запускается через pre-commit.

Запуск вручную:
```
pixi run bandit -r src
```

---

## 2.4 Создание конфигурационных файлов

В проекте созданы:

- `pyproject.toml` — конфигурации всех инструментов  
- `pixi.toml` — управление зависимостями  
- `.pre-commit-config.yaml` — хуки  
- `.gitignore`  
- `.gitattributes`  

---

# Управление зависимостями

## 3.1 Пакетный менеджер Pixi

Pixi используется как основной менеджер зависимостей.

Добавление зависимостей:

```
pixi add numpy pandas scikit-learn seaborn matplotlib
pixi add black isort ruff mypy bandit pre-commit
```

Pixi создаёт:

- `pixi.toml` — список зависимостей  
- `pixi.lock` — зафиксированное окружение  

---

## 3.2 pyproject с конфигурациями

`pyproject.toml` содержит конфиги для:

- Black  
- isort  
- Ruff  
- MyPy  
- Bandit  

---

## 3.3 Настройка виртуального окружения

Pixi автоматически создаёт среду.

Проверка:

```
pixi run python --version
```

---

## 3.4 Dockerfile для контейнеризации

Создан минимальный Dockerfile:

```dockerfile
FROM python:3.11-slim

RUN pip install pixi

WORKDIR /app

COPY pixi.toml pixi.lock ./
COPY pyproject.toml ./
COPY src ./src

RUN pixi install
```

---

# Git workflow (1 балл)

## 4.1 Настройка репозитория
Создан репозиторий

---

## 4.2 .gitignore

Использован `.gitignore`, включающий:

- временные файлы  
- окружения  
- данные  
- артефакты ML  
- кэш  
- модели  

---

## 4.3 Ветки

Созданы ветки:

```
main — основная ветка
hw1  — выполнение ДЗ 1
dev  — ветка разработки
```

---

# Заключение

В ходе выполнения работы было настроено полноценное ML-инженерное окружение:

- создан шаблон Cookiecutter  
- настроено качество кода (Black, Ruff, isort, MyPy, Bandit)  
- настроен пакетный менеджер Pixi  
- создан Dockerfile  
- организован Git workflow  
- оформлен отчёт
