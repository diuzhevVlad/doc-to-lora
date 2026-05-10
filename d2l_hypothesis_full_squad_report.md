# Финальный отчет: D2L Hypothesis на SQuAD

## Краткий вывод

Проведен полный эксперимент D2L Hypothesis на SQuAD validation: `10,570` примеров для каждого условия. Теперь закрыты оба типа сравнений: D2L adapter conditions, wrong-adapter controls, no-context lower bounds и visible-prompt upper bound.

Главный результат: D2L действительно переносит скрытую информацию через adapter parameters, но качество переноса ограничено. Hidden question улучшает QA F1 с `0.098` до `0.189`, а hidden instruction улучшает format compliance с `0.0046` до `0.354`. Visible prompting остается значительно сильнее: `0.698` QA F1 и `0.894` compliance.

Итоговая формулировка: текущий D2L checkpoint показывает **measurable but weak internalization**. Это полезный положительный сигнал для research narrative, но не доказательство, что D2L надежно заменяет prompt или system instruction.

## Исследовательский вопрос

Doc-to-LoRA обучает гиперсеть, которая по текстовому контексту генерирует LoRA-адаптер для base model. В статье это подается как способ быстро internalize'ить контекст и затем отвечать без повторного добавления длинного документа в prompt.

В этом проекте проверяется более узкий вопрос:

> Что именно internalize'ит D2L: содержательную информацию или инструкцию поведения?

Для проверки QA-запрос разделен на две части:

- `content` - сам вопрос, то есть что нужно найти в passage;
- `instruction` - требование к формату ответа, например answer only или full sentence.

## Дизайн эксперимента

| condition | Adapter context | Visible prompt | Что проверяет |
|---|---|---|---|
| `content_adapter` | правильный вопрос | passage + instruction | перенос hidden question |
| `content_wrong` | вопрос другого примера | passage + instruction | negative control для content |
| `instruction_adapter` | правильная инструкция | passage + question | перенос hidden instruction |
| `instruction_wrong` | противоположная инструкция | passage + question | negative control для instruction |

Wrong conditions построены детерминированно через circular shift by one example. Это делает контроль воспроизводимым.

Base-model controls:

| control | Adapter context | Visible prompt | Роль |
|---|---|---|---|
| `prompt_only_no_context` | пусто | passage + question + instruction | upper bound |
| `content_adapter_no_context` | пусто | passage + instruction | lower bound для content |
| `instruction_adapter_no_context` | пусто | passage + question | lower bound для instruction |

## Данные и артефакты

Dataset: SQuAD v1.1 validation.

Размер: `10,570` примеров на condition.

D2L checkpoint:

```text
trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
```

D2L output:

```text
trained_d2l/qwen_4b_d2l/eval-results-20000/20260508-122632_0f46e0ac/
```

Control output:

```text
eval_results/Qwen/Qwen3-4B-Instruct-2507/20260509-230150_f4e0af7f/
```

Итоговые CSV:

```text
trained_d2l/qwen_4b_d2l/eval-results-20000/20260508-122632_0f46e0ac/evaluation_results_generation.csv
eval_results/Qwen/Qwen3-4B-Instruct-2507/20260509-230150_f4e0af7f/evaluation_results_generation_no_context.csv
```

Per-sample outputs сохранены в `*_generated_text.jsonl`, поэтому можно делать ручной audit.

## Метрики

`qa_f1_score` - overlap generated answer с gold answer. Это основная метрика для content transfer.

`qa_precision` - какая доля сгенерированного ответа совпадает с gold answer.

`qa_recall` - насколько gold answer покрыт сгенерированным ответом.

`format_compliance` - доля ответов, которые соблюдают требуемый формат. В текущей версии проверяются два формата: `answer_only` и `full_sentence`.

`correctness_given_compliant` - QA F1 только среди compliant examples. Эту метрику нужно читать вместе с `compliant N`: если compliant examples мало, оценка нестабильна.

## Основная таблица результатов

| run | N | compliant N | qa_f1 | precision | recall | format compliance | correctness given compliant |
|---|---:|---:|---:|---:|---:|---:|---:|
| `content_adapter_no_context` | 10,570 | 4,817 | 0.098 | 0.064 | 0.451 | 0.456 | 0.105 |
| `content_wrong` | 10,570 | 3,285 | 0.061 | 0.037 | 0.369 | 0.311 | 0.067 |
| `content_adapter` | 10,570 | 3,920 | 0.189 | 0.125 | 0.750 | 0.371 | 0.216 |
| `instruction_adapter_no_context` | 10,570 | 49 | 0.309 | 0.202 | 0.946 | 0.0046 | 0.605 |
| `instruction_wrong` | 10,570 | 286 | 0.566 | 0.491 | 0.925 | 0.027 | 0.823 |
| `instruction_adapter` | 10,570 | 3,741 | 0.568 | 0.495 | 0.921 | 0.354 | 0.897 |
| `prompt_only_no_context` | 10,570 | 9,447 | 0.698 | 0.615 | 0.931 | 0.894 | 0.731 |

## Content transfer

Content setup проверяет, может ли D2L перенести скрытый вопрос. В prompt видны passage и instruction, но вопрос скрыт в adapter context.

| run | Смысл | qa_f1 |
|---|---|---:|
| `content_adapter_no_context` | вопроса нет ни в prompt, ни в adapter | 0.098 |
| `content_wrong` | в adapter неправильный вопрос | 0.061 |
| `content_adapter` | в adapter правильный вопрос | 0.189 |
| `prompt_only_no_context` | вопрос явно виден в prompt | 0.698 |

Правильный hidden question дает:

- `+0.090` QA F1 относительно no-question lower bound;
- `+0.127` QA F1 относительно wrong-adapter;
- `1.92x` относительно no-question lower bound;
- `3.07x` относительно wrong-adapter.

Доля закрытого разрыва между lower bound и visible-prompt upper bound:

```text
(0.189 - 0.098) / (0.698 - 0.098) ~= 15%
```

Интерпретация: D2L переносит часть содержательного сигнала вопроса. Но перенос слабый: visible question в prompt остается намного надежнее.

Отдельно важно, что `content_wrong` хуже, чем `content_adapter_no_context`. Неправильный adapter не просто бесполезен, он может уводить модель в неправильный режим поиска.

## Instruction transfer

Instruction setup проверяет, может ли D2L перенести скрытую форматную инструкцию. В prompt видны passage и question, но инструкция скрыта в adapter context.

| run | Смысл | format compliance |
|---|---|---:|
| `instruction_adapter_no_context` | инструкции нет | 0.0046 |
| `instruction_wrong` | неправильная hidden instruction | 0.027 |
| `instruction_adapter` | правильная hidden instruction | 0.354 |
| `prompt_only_no_context` | инструкция явно видна в prompt | 0.894 |

Правильная hidden instruction дает:

- `+0.349` compliance относительно no-instruction lower bound;
- `+0.327` compliance относительно wrong instruction;
- `76.3x` относительно no-instruction lower bound;
- `13.1x` относительно wrong instruction.

Доля закрытого разрыва между no-instruction lower bound и visible-prompt upper bound:

```text
(0.354 - 0.0046) / (0.894 - 0.0046) ~= 39%
```

Интерпретация: простая форматная policy действительно частично переносится через D2L adapter. На этом setup instruction transfer выглядит сильнее по gap closure, чем content transfer.

Но absolute compliance остается умеренным: `0.354` означает, что около двух третей ответов все еще не соблюдают нужный формат. Поэтому вывод должен быть именно про partial instruction internalization.

## Почему QA F1 в instruction conditions нужно читать осторожно

В `instruction_adapter`, `instruction_wrong` и `instruction_adapter_no_context` вопрос виден в prompt. Поэтому QA F1 там не является чистой метрикой instruction transfer.

Например:

- `instruction_adapter`: `qa_f1=0.568`;
- `instruction_wrong`: `qa_f1=0.566`;
- `instruction_adapter_no_context`: `qa_f1=0.309`.

Главная метрика для instruction transfer - `format_compliance`. QA F1 меняется из-за длины и формы ответа: без инструкции модель часто генерирует длинный ответ с высоким recall (`0.946`), но низкой precision (`0.202`), поэтому F1 ниже.

## Связь с исходной гипотезой

Начальная гипотеза была: D2L лучше переносит content, чем instruction.

Финальные результаты ее не подтверждают в сильной форме. Получилась более интересная картина:

- content transfer есть, но закрывает только около `15%` gap до visible prompt;
- instruction transfer тоже есть и закрывает около `39%` gap до visible prompt;
- оба переноса заметно уступают visible prompting.

Поэтому корректный новый вывод:

> D2L способен частично internalize'ить и hidden question content, и простую hidden format instruction. В данном SQuAD setup instruction compliance переносится даже более заметно относительно своего lower bound, но абсолютное качество обоих переносов остается недостаточным для практической замены prompt.

## Runtime

D2L-прогон четырех adapter conditions:

| condition | runtime | samples/sec |
|---|---:|---:|
| `content_adapter` | 8:35:40 | 0.342 |
| `content_wrong` | 9:23:29 | 0.313 |
| `instruction_adapter` | 3:32:41 | 0.828 |
| `instruction_wrong` | 3:36:22 | 0.814 |

Base-model controls:

| control | runtime | samples/sec |
|---|---:|---:|
| `prompt_only_no_context` | 1:58:43 | 1.484 |
| `content_adapter_no_context` | 6:43:48 | 0.436 |
| `instruction_adapter_no_context` | 5:08:58 | 0.570 |

Control session завершилась успешно:

```text
FINISHED_AT=2026-05-10T14:26:02+03:00 EXIT_STATUS=0
```

## Ограничения

Эксперимент пока только на SQuAD. Для более сильного утверждения нужно повторить setup на ROPES и QASPER.

Instruction transfer v1 проверяет только простые форматные инструкции: `answer_only` и `full_sentence`. Это не доказывает перенос сложных policies, отказов, reasoning instructions или stable persona/system behavior.

`format_compliance` - эвристическая метрика. Она достаточна для сравнения условий, но borderline outputs стоит проверить вручную.

Использовался существующий D2L checkpoint, не обученный специально под hidden question / hidden instruction objective. Поэтому отрицательная часть результата относится к текущему checkpoint/setup, а не к принципиальному пределу архитектуры.

## Что можно утверждать

Можно утверждать:

- D2L переносит hidden question content лучше, чем no-question и wrong-adapter baselines.
- D2L переносит hidden answer-format instruction лучше, чем no-instruction и wrong-instruction baselines.
- Эффект подтвержден на полном SQuAD validation, а не только на 100-примерном pilot.
- Visible prompting остается намного сильнее D2L internalization.
- Internalization измерима, но неполная.

Не стоит утверждать:

- что D2L лучше visible prompting;
- что D2L надежно заменяет prompt или system instruction;
- что D2L уже является robust behavior adapter;
- что результат автоматически переносится на ROPES, QASPER или другие QA datasets.

## Следующие шаги

1. Разбить `format_compliance` по `instruction_type`: отдельно `answer_only` и `full_sentence`.

2. Сделать ручной audit generated outputs: 30-50 примеров на condition.

3. Повторить эксперимент на ROPES и QASPER, если нужен более сильный финальный отчет.

4. Если цель - улучшить качество, обучить или дообучить D2L checkpoint именно на hidden question / hidden instruction objective.

5. Для статьи/доклада сформулировать результат как boundary analysis: D2L может переносить скрытые сигналы, но текущий checkpoint делает это частично и заметно уступает обычному prompt.

## Финальная формулировка

Полный SQuAD эксперимент показывает, что D2L действительно передает часть скрытого сигнала через adapter parameters. Правильный hidden question повышает QA F1, а правильная hidden instruction повышает format compliance.

При этом результат не подтверждает простую версию гипотезы "content переносится лучше instruction". Наоборот, в этой постановке instruction compliance закрывает большую долю разрыва до visible prompting. Более точный вывод: D2L partial-internalizes both content and simple instructions, but neither transfer is robust enough to replace explicit prompting.

