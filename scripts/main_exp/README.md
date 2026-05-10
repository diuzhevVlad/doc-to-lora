# Основной D2L pipeline

Эта папка содержит скрипты для воспроизведения основного pipeline из статьи Doc-to-LoRA: подготовка данных, training и evaluation.

## Данные

Можно скачать уже сгенерированные данные или сгенерировать их заново. Скачивание обычно практичнее: полный набор для всех моделей занимает около `328 GB`.

```bash
uv run bash scripts/main_exp/0-download_data.sh
```

Генерация с нуля может занять очень много времени, особенно без параллелизации по нескольким GPU.

```bash
# optional: generate training data from scratch
# uv run bash scripts/main_exp/gen_data.sh
```

## Обучение

После подготовки данных запустить training:

```bash
uv run bash scripts/main_exp/1-train.sh
```

## Оценка

Скрипты для воспроизведения основных результатов статьи лежат в:

```text
scripts/main_exp/eval/
```
