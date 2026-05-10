# Полный отчет: D2L Hypothesis на SQuAD

## Краткий вывод

Проведен полный D2L-прогон на SQuAD validation: `10,570` примеров для каждого из четырех adapter conditions. Результаты показывают measurable but weak internalization: правильный скрытый вопрос улучшает QA F1, а правильная скрытая инструкция улучшает format compliance.

Сильная формулировка была бы неверной: D2L не заменяет visible prompting. На полном SQuAD visible prompt дает `qa_f1=0.698`, тогда как D2L с правильным hidden question дает `0.189`.

## Исследовательский вопрос

Doc-to-LoRA обучает гиперсеть, которая по текстовому контексту генерирует LoRA-адаптер для base model. В статье это подается как способ быстро internalize'ить контекст.

В этом проекте проверяется более узкий вопрос:

> Что именно internalize'ит D2L: содержательную информацию или инструкцию поведения?

Для этого QA-запрос разделен на две части:

- `content` - сам вопрос, то есть что нужно найти в passage;
- `instruction` - требование к формату ответа, например answer only или full sentence.

## Дизайн эксперимента

| condition | Adapter context | Visible prompt | Что проверяет |
|---|---|---|---|
| `content_adapter` | правильный вопрос | passage + instruction | перенос hidden question |
| `content_wrong` | вопрос другого примера | passage + instruction | negative control для content |
| `instruction_adapter` | правильная инструкция | passage + question | перенос hidden instruction |
| `instruction_wrong` | противоположная инструкция | passage + question | negative control для instruction |

Wrong conditions построены детерминированно через circular shift by one example. Это делает контроль воспроизводимым и исключает случайную выборку неправильного контекста.

Отдельно запущены base-model controls:

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

Полный D2L output:

```text
trained_d2l/qwen_4b_d2l/eval-results-20000/20260508-122632_0f46e0ac/
```

Control output:

```text
eval_results/Qwen/Qwen3-4B-Instruct-2507/20260509-230150_f4e0af7f/
```

Сгенерированные ответы сохранены в `*_generated_text.jsonl`, поэтому можно делать ручной audit отдельных примеров.

## Метрики

`qa_f1_score` - overlap generated answer с gold answer. Это основная метрика для content transfer.

`qa_precision` - какая доля сгенерированного ответа совпадает с gold answer.

`qa_recall` - насколько gold answer покрыт сгенерированным ответом.

`format_compliance` - доля ответов, которые соблюдают требуемый формат. В текущей версии проверяются два формата: `answer_only` и `full_sentence`.

`correctness_given_compliant` - QA F1 только среди compliant examples. Эту метрику нельзя читать без `compliant N`: если compliant examples мало, оценка нестабильна.

## Основные D2L результаты

| condition | N | compliant N | qa_f1 | precision | recall | format compliance | correctness given compliant |
|---|---:|---:|---:|---:|---:|---:|---:|
| `content_adapter` | 10,570 | 3,920 | 0.189 | 0.125 | 0.750 | 0.371 | 0.216 |
| `content_wrong` | 10,570 | 3,285 | 0.061 | 0.037 | 0.369 | 0.311 | 0.067 |
| `instruction_adapter` | 10,570 | 3,741 | 0.568 | 0.495 | 0.921 | 0.354 | 0.897 |
| `instruction_wrong` | 10,570 | 286 | 0.566 | 0.491 | 0.925 | 0.027 | 0.823 |

## Base-model controls

| control | N | compliant N | qa_f1 | precision | recall | format compliance | correctness given compliant |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_only_no_context` | 10,570 | 9,447 | 0.698 | 0.615 | 0.931 | 0.894 | 0.731 |
| `content_adapter_no_context` | 10,570 | 4,817 | 0.098 | 0.064 | 0.451 | 0.456 | 0.105 |
| `instruction_adapter_no_context` | in progress | in progress | in progress | in progress | in progress | in progress | in progress |

`instruction_adapter_no_context` еще выполняется в `tmux` session `d2l_squad_controls`. После завершения нужно обновить эту таблицу и финальное сравнение по instruction transfer.

## Сравнение content transfer

| run | Смысл | qa_f1 |
|---|---|---:|
| `content_adapter_no_context` | нет вопроса ни в prompt, ни в adapter | 0.098 |
| `content_wrong` | в adapter неправильный вопрос | 0.061 |
| `content_adapter` | в adapter правильный вопрос | 0.189 |
| `prompt_only_no_context` | вопрос явно виден в prompt | 0.698 |

Вывод: правильный hidden question реально влияет на ответ. D2L дает `+0.090` QA F1 относительно no-question lower bound и `+0.127` относительно wrong-adapter.

Но эффект слабый в абсолютном выражении. Доля закрытого разрыва между lower bound и upper bound:

```text
(0.189 - 0.098) / (0.698 - 0.098) ~= 15%
```

Это хороший сигнал для исследовательского отчета, но плохой результат для практической замены visible question.

## Сравнение instruction transfer

| run | Смысл | format compliance |
|---|---|---:|
| `instruction_wrong` | неправильная hidden instruction | 0.027 |
| `instruction_adapter` | правильная hidden instruction | 0.354 |
| `prompt_only_no_context` | инструкция явно видна в prompt | 0.894 |
| `instruction_adapter_no_context` | инструкции нет | in progress |

Вывод: D2L переносит простую форматную инструкцию. Compliance растет с `0.027` до `0.354`, то есть эффект относительно wrong instruction большой.

Однако visible prompt намного сильнее: `0.894` compliance. Поэтому корректная интерпретация - partial instruction internalization, а не robust policy transfer.

## Почему QA F1 высокий в instruction conditions

В `instruction_adapter` и `instruction_wrong` вопрос виден в prompt. Поэтому QA F1 там отражает в основном способность модели отвечать на SQuAD-вопрос, а не способность internalize'ить instruction.

Именно поэтому `qa_f1` почти одинаковый:

- `instruction_adapter`: `0.568`;
- `instruction_wrong`: `0.566`.

Главная метрика для instruction transfer - `format_compliance`, а не QA F1.

## Связь с pilot

Pilot на 100 примерах показывал похожее направление:

| группа | D2L correct | wrong | no-context | visible prompt |
|---|---:|---:|---:|---:|
| content QA F1 | 0.198 | 0.052 | 0.074 | 0.613 |
| instruction compliance | 0.31 | 0.03 | 0.00 | 0.72 |

Full SQuAD подтвердил главное направление на большем числе примеров:

- content correct adapter лучше wrong adapter;
- instruction correct adapter лучше wrong adapter;
- visible prompting остается верхней границей.

## Runtime

D2L-прогон четырех conditions:

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
| `instruction_adapter_no_context` | in progress | in progress |

Content conditions медленнее instruction conditions, потому что prompt/answer dynamics чаще приводят к более долгой генерации. Сам D2L overhead есть, но основное время уходит в autoregressive generation.

## Ограничения

Эксперимент пока только на SQuAD. Для более сильного утверждения нужно повторить на ROPES и QASPER.

Instruction transfer v1 проверяет только простые форматные инструкции: `answer_only` и `full_sentence`. Это не доказывает перенос сложных policies.

`format_compliance` - эвристическая метрика. Она достаточна для автоматического сравнения условий, но часть borderline outputs лучше проверять вручную.

Control `instruction_adapter_no_context` еще не завершен, поэтому нижняя граница instruction transfer пока не закрыта полностью.

## Что можно утверждать

Можно утверждать:

- D2L переносит hidden question content лучше, чем wrong adapter.
- D2L переносит hidden answer-format instruction лучше, чем wrong instruction adapter.
- Эффект подтверждается на полном SQuAD validation, а не только на pilot.
- Internalization измерима, но слабая.

Не стоит утверждать:

- что D2L лучше visible prompting;
- что D2L надежно заменяет prompt;
- что D2L уже является полноценным behavior adapter;
- что результат автоматически переносится на ROPES, QASPER или другие QA datasets.

## Следующие шаги

1. Дождаться `instruction_adapter_no_context` и обновить таблицу controls.

2. Разбить результаты по `instruction_type`: `answer_only` против `full_sentence`.

3. Сделать ручной audit generated outputs: 30-50 примеров на condition.

4. Повторить эксперимент на ROPES и QASPER для stronger external validity.

5. Если цель - улучшить качество, обучить или дообучить D2L checkpoint именно на hidden question / hidden instruction objective.

## Финальная формулировка

Полный SQuAD эксперимент показывает, что D2L действительно передает часть скрытого сигнала через adapter parameters. Правильный hidden question повышает QA F1, а правильная hidden instruction повышает format compliance.

Но оба эффекта остаются далеко от visible prompting. Поэтому результат лучше формулировать как evidence of partial internalization, а не как proof of robust hidden-context transfer.

