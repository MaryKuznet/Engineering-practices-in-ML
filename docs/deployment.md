# Развёртывание проекта

В этом разделе описаны способы развёртывания и запуска проекта.

---

## Локальный запуск (рекомендуется)

### Требования
- Python >= 3.11
- Pixi
- Git

### Установка

```bash
git clone <repository_url>
cd <repository_name>
pixi install
```

---

## Загрузка данных

Проект использует DVC для управления данными:

```bash
pixi run dvc pull
```

---

## Запуск обучения

```bash
pixi run train
```

---

## Использование Docker (опционально)

Если проект запускается в Docker-контейнере:

```bash
docker build -t ml-project .
docker run --rm ml-project
```

---

## Примечания

- Для MLflow убедитесь, что корректно настроен `tracking_uri`
- Для ClearML необходимо наличие файла `~/.clearml.conf`
