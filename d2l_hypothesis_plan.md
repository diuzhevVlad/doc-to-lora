# План и статус эксперимента D2L Hypothesis

## Коротко

Цель проекта - проверить, что именно переносит Doc-to-LoRA через сгенерированный LoRA-адаптер: содержательную информацию из запроса или инструкцию о формате ответа. Мы разделили эти два сигнала на SQuAD и прогнали D2L checkpoint на полном validation split.

Текущий вывод: D2L показывает измеримую, но слабую internalization. Правильный скрытый вопрос улучшает QA F1 относительно wrong-adapter и no-context baseline, а правильная скрытая инструкция резко повышает format compliance относительно wrong instruction. При этом visible prompting остается намного сильнее.

## Мотивация

Doc-to-LoRA предлагает один раз "прочитать" контекст и превратить его в LoRA-параметры, чтобы затем отвечать без повторного добавления длинного документа в prompt. Это может снижать latency, размер prompt и KV-cache при repeated inference.

Для практики важно понять границу метода. Если D2L переносит в основном факты и содержательный контекст, то это механизм document memory. Если он также переносит инструкции и policy, то его можно рассматривать как основу для переключаемых behavioral adapters.

## Гипотеза

Основная гипотеза:

> D2L лучше internalize'ит содержательную часть запроса, чем инструкционную часть.

В терминах эксперимента:

- `content transfer` - модель должна восстановить, о чем спрашивают, если вопрос скрыт в adapter context;
- `instruction transfer` - модель должна соблюсти формат ответа, если инструкция скрыта в adapter context.

## Почему это следует проверять отдельно

В query-internalization setup из статьи вопрос часто содержит сразу два типа сигнала: что искать и как отвечать. Например, `What team did the Panthers defeat?` задает content, а `Output only the answer` задает instruction.

Если не разделить эти сигналы, нельзя понять, что именно D2L перенес в параметры. Поэтому эксперимент устроен так, чтобы content и instruction измерялись отдельно.

## Данные

Используется SQuAD v1.1 validation:

- полный split: `10,570` примеров;
- pilot split: первые `100` примеров для быстрой проверки;
- deterministic seed: `42`;
- wrong-adapter examples строятся circular shift by one example.

Сгенерированные данные лежат здесь:

```text
data/raw_datasets/d2l_hypothesis/squad/<condition>/validation.jsonl
```

Каждая строка содержит нормализованные поля:

- `context` - текст, который подается в D2L adapter context;
- `prompts` - visible prompt для base model;
- `responses` - gold answer;
- `condition`, `instruction`, `question`, `gold_answer` и другие metadata.

## Условия

| condition | Adapter context | Visible prompt | Назначение |
|---|---|---|---|
| `content_adapter` | правильный вопрос | passage + instruction | главный тест content transfer |
| `content_wrong` | вопрос другого примера | passage + instruction | negative control для content |
| `instruction_adapter` | правильная инструкция | passage + question | главный тест instruction transfer |
| `instruction_wrong` | противоположная инструкция | passage + question | negative control для instruction |
| `prompt_only` | пусто | passage + question + instruction | upper bound visible prompting |

Для lower-bound controls используется base model без D2L context:

- `content_adapter_no_context`: passage + instruction, но без скрытого вопроса;
- `instruction_adapter_no_context`: passage + question, но без скрытой инструкции;
- `prompt_only_no_context`: весь passage + question + instruction виден в prompt.

## Что было реализовано

Добавлен builder данных:

```text
data/build_d2l_hypothesis_squad.py
```

Он строит все D2L hypothesis conditions из SQuAD и сохраняет их в `data/raw_datasets/d2l_hypothesis/squad/`.

В data pipeline добавлена поддержка этих уже подготовленных JSONL-файлов:

- регистрация датасетов в `src/ctx_to_lora/data/definitions.py`;
- сохранение metadata в `src/ctx_to_lora/data/processing.py`;
- минимальная обработка already-processed данных в `src/ctx_to_lora/data/preprocessing_fn.py`;
- удаление metadata перед batch collation в `src/ctx_to_lora/data/collator.py`.

В evaluation добавлены метрики:

- `format_compliance`;
- `correctness_given_compliant`;
- сохранение metadata рядом с generated text для ручного анализа.

В `src/ctx_to_lora/modeling/hypernet.py` оставлен небольшой memory cleanup перед загрузкой checkpoint. Он нужен на RTX 3080 Laptop 16 GB, иначе повторная инициализация D2L checkpoint может падать с CUDA OOM.

## Метрики

`qa_f1_score` - overlap между generated answer и gold answer. Это главная метрика для content transfer.

`qa_precision` и `qa_recall` показывают, насколько ответ точен и насколько он покрывает gold answer. В этих прогонах recall часто высокий, а precision ниже: модель часто содержит правильные слова, но добавляет лишний текст.

`format_compliance` проверяет, соблюден ли требуемый формат. В v1 есть два типа инструкций: `answer_only` и `full_sentence`.

`correctness_given_compliant` считает QA F1 только по тем примерам, где формат соблюден. Эту метрику нужно читать вместе с количеством compliant examples.

## Полный D2L Run

Полный прогон четырех D2L conditions завершен.

```text
trained_d2l/qwen_4b_d2l/eval-results-20000/20260508-122632_0f46e0ac/
```

| condition | N | compliant N | qa_f1 | precision | recall | compliance | correctness given compliant |
|---|---:|---:|---:|---:|---:|---:|---:|
| `content_adapter` | 10,570 | 3,920 | 0.189 | 0.125 | 0.750 | 0.371 | 0.216 |
| `content_wrong` | 10,570 | 3,285 | 0.061 | 0.037 | 0.369 | 0.311 | 0.067 |
| `instruction_adapter` | 10,570 | 3,741 | 0.568 | 0.495 | 0.921 | 0.354 | 0.897 |
| `instruction_wrong` | 10,570 | 286 | 0.566 | 0.491 | 0.925 | 0.027 | 0.823 |

Главные сравнения:

- content: `content_adapter` лучше `content_wrong` на `+0.127` QA F1;
- instruction: `instruction_adapter` лучше `instruction_wrong` на `+0.327` compliance;
- QA F1 почти одинаковый у `instruction_adapter` и `instruction_wrong`, потому что вопрос виден в prompt в обоих условиях.

## Full SQuAD Controls

Контрольный base-model прогон запущен в `tmux`:

```text
d2l_squad_controls
```

Готовые результаты:

```text
eval_results/Qwen/Qwen3-4B-Instruct-2507/20260509-230150_f4e0af7f/
```

| control | N | compliant N | qa_f1 | precision | recall | compliance | correctness given compliant |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_only_no_context` | 10,570 | 9,447 | 0.698 | 0.615 | 0.931 | 0.894 | 0.731 |
| `content_adapter_no_context` | 10,570 | 4,817 | 0.098 | 0.064 | 0.451 | 0.456 | 0.105 |
| `instruction_adapter_no_context` | in progress | in progress | in progress | in progress | in progress | in progress | in progress |

Для content transfer уже можно поставить D2L между lower и upper bound:

| run | visible information | qa_f1 |
|---|---|---:|
| no hidden question | passage + instruction | 0.098 |
| wrong hidden question | passage + instruction + wrong adapter | 0.061 |
| correct hidden question via D2L | passage + instruction + correct adapter | 0.189 |
| visible full prompt | passage + question + instruction | 0.698 |

D2L закрывает примерно `15%` разрыва между no-question lower bound и visible-prompt upper bound:

```text
(0.189 - 0.098) / (0.698 - 0.098) ~= 0.15
```

## Интерпретация

Content transfer работает измеримо: правильный скрытый вопрос дает лучший ответ, чем wrong question и no-context control. Но абсолютный результат низкий: `0.189` против `0.698` у visible prompting.

Instruction transfer тоже работает измеримо: compliance растет с `0.027` при wrong instruction до `0.354` при correct hidden instruction. Это значит, что простая форматная policy частично переносится через D2L adapter.

Итоговая формулировка должна быть аккуратной: текущий checkpoint показывает measurable but weak internalization. Это не доказательство надежной замены prompt'а и не доказательство, что D2L robustly переносит сложные policies.

## Что еще нужно доделать

1. Дождаться `instruction_adapter_no_context`, чтобы закрыть lower bound для instruction transfer.

2. Разбить результаты по `instruction_type`: отдельно `answer_only` и `full_sentence`.

3. Сделать ручной audit generated outputs: по 30-50 примеров на condition.

4. Повторить эксперимент на ROPES и QASPER, если нужен более сильный итоговый отчет.

5. Для улучшения качества попробовать checkpoint, специально обученный на hidden question / hidden instruction setup.

## Артефакты

Основные документы:

- `d2l_hypothesis_narrative.md` - короткий нарратив для коллег и доклада;
- `d2l_hypothesis_full_squad_report.md` - полный отчет по SQuAD;
- `d2l_flash_attention_env.md` - окружение, команды запуска и runtime.

Основные outputs:

```text
trained_d2l/qwen_4b_d2l/eval-results-20000/20260508-122632_0f46e0ac/evaluation_results_generation.csv
eval_results/Qwen/Qwen3-4B-Instruct-2507/20260509-230150_f4e0af7f/evaluation_results_generation_no_context.csv
```

