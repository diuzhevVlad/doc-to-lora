# Viewer для self-generated данных

Этот viewer нужен для ручного просмотра self-generated датасетов: контекстов, вопросов и ответов.

## Запуск

```bash
uv run webui/self_gen_viewer.py
```

После запуска открыть в браузере:

```text
http://localhost:5001
```

## Как пользоваться

1. Выбрать папку модели в dropdown.
2. Выбрать parquet-файл из доступного списка.
3. Задать число примеров для просмотра.
4. Нажать `Load Data` и посмотреть контекст с QA-парами.

## Ожидаемая структура данных

```text
data/raw_datasets/self_gen/
├── google/
│   └── gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0/
│       └── fw_qa_v2/
│           └── *.parquet
└── mistralai/
    └── Mistral-7B-Instruct-v0.2_temp_0.0_closed_qa_prob_1.0/
        └── *.parquet
```
