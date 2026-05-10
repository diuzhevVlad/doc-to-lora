# Нарратив проекта: что internalize'ит Doc-to-LoRA

## 1. Идея

Doc-to-LoRA превращает входной контекст в LoRA-адаптер: модель как будто один раз "читает" документ, а затем отвечает без повторного добавления этого документа в prompt. Это интересно для repeated inference, потому что длинный prompt увеличивает latency, память KV-cache и стоимость запросов.

Главный практический вопрос: D2L хранит только факты и содержательный контекст или может также переносить поведение модели, например формат ответа?

## 2. Гипотеза

Гипотеза проекта: D2L лучше переносит **content**, чем **instruction**.

То есть сигнал "что искать" должен internalize'иться надежнее, чем сигнал "как отвечать".

## 3. Эксперимент

Мы разделили исходный QA-запрос на две части: содержательный вопрос и форматную инструкцию. Затем построили SQuAD conditions, где в adapter context скрывается либо вопрос, либо инструкция.

Для проверки добавлены wrong-adapter controls: adapter получает вопрос или инструкцию от другого примера. Также используются base-model controls: полный visible prompt как upper bound и prompt без скрытого сигнала как lower bound.

## 4. Результаты: content transfer

| Условие | Что видит модель | QA F1 |
|---|---|---:|
| No hidden question | passage + instruction | 0.098 |
| Wrong hidden question | passage + instruction + wrong adapter | 0.061 |
| D2L correct hidden question | passage + instruction + correct adapter | 0.189 |
| Visible full prompt | passage + question + instruction | 0.698 |

Правильный hidden question через D2L заметно лучше wrong-adapter и no-question baseline. Значит, D2L действительно переносит часть содержательного вопроса.

Но качество остается слабым: D2L закрывает только около `15%` разрыва между lower bound и visible prompting.

## 5. Результаты: instruction transfer

| Условие | Что измеряем | Format compliance |
|---|---|---:|
| Wrong hidden instruction | неверная инструкция в adapter | 0.027 |
| D2L correct hidden instruction | правильная инструкция в adapter | 0.354 |
| Visible full prompt | инструкция явно в prompt | 0.894 |
| No hidden instruction | base model без инструкции | in progress |

Правильная hidden instruction сильно повышает compliance относительно wrong instruction. Значит, D2L переносит не только содержание, но и простую форматную policy.

Однако абсолютный уровень compliance низкий: примерно две трети ответов все еще не соблюдают формат. Поэтому это partial instruction internalization, а не надежная замена system prompt.

## 6. Что из этого следует

Результаты поддерживают аккуратную версию гипотезы: D2L переносит скрытую информацию, но делает это неполно. Content transfer измерим, instruction transfer тоже есть, но оба результата сильно уступают visible prompting.

Практически текущий D2L checkpoint лучше описывать как weak context/query memory. Его пока нельзя уверенно использовать как robust behavior adapter для устойчивого переноса инструкций.

## 7. Что делать дальше

Сначала нужно дождаться последнего full SQuAD control: `instruction_adapter_no_context`. После этого сравнение content и instruction будет закрыто нижними и верхними контролями.

Затем стоит разобрать результаты по типам инструкций (`answer_only` и `full_sentence`) и вручную посмотреть generated outputs. Если цель - сильный итоговый отчет, следующий шаг - повторить setup на ROPES и QASPER.

