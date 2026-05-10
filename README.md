<div align="center">
    <h1>Doc-to-LoRA (D2L): Learning to Instantly Internalize Contexts</h1>
    <a href="https://pub.sakana.ai/doc-to-lora/">Interactive Web</a> |
    <a href="https://arxiv.org/abs/2602.15902">Paper</a> |
    <a href="https://huggingface.co/SakanaAI">Hugging Face</a> |
    <a href="https://github.com/SakanaAI/doc-to-lora">GitHub</a>
<br>Репозиторий с reference implementation Doc-to-LoRA и локальными экспериментами по D2L Hypothesis.<br>
</div>

<div align="center">
    <img height="300px" src="assets/overview_animation.gif" />
</div>

---

## Локальный эксперимент D2L Hypothesis

В этом форке добавлен эксперимент, который проверяет, что D2L переносит через LoRA-адаптер: содержательный вопрос или инструкцию о формате ответа.

Основные документы:

| Файл | Что внутри |
|---|---|
| `d2l_hypothesis_narrative.md` | короткий русский нарратив для коллег и доклада |
| `d2l_hypothesis_plan.md` | мотивация, дизайн, условия, статус и план следующих шагов |
| `d2l_hypothesis_full_squad_report.md` | полный отчет по SQuAD с таблицами результатов |
| `d2l_flash_attention_env.md` | conda-окружение, команды запуска и runtime |

Короткий вывод эксперимента: D2L показывает measurable but weak internalization. Правильный hidden question повышает QA F1, а правильная hidden instruction повышает format compliance, но visible prompting остается намного сильнее.

## Установка

Базовая установка исходного проекта:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
./install.sh
```

Для локальных SQuAD-прогонов использовалась отдельная `conda`-среда `d2l-flash`; команды описаны в `d2l_flash_attention_env.md`.

## Предобученные модели

```bash
uv run huggingface-cli login
uv run huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l --include "*/"
```

## API Python

```python
# Интерфейс ниже поддерживает только single-example inference.
# Для batched inference см. src/ctx_to_lora/modeling/hypernet.py.
import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

doc = open("data/sakana_wiki.txt", "r").read()
chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

model.internalize(doc)

outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))

# Чтобы убрать internalized context:
# model.reset()
```

## Интерактивная демо-страница

```bash
uv run demo/app.py
```

<div align="center">
    <h3>Video Demo</h3>
    <video src="https://github.com/user-attachments/assets/16781365-5ec2-4c1c-b4f4-aeeebe3c2be5" controls autoplay muted playsinline preload="metadata" width="900"></video>
</div>

## Экспериментальные скрипты

Команды запускаются из корня проекта через `uv run`.

| Эксперимент | Подготовка данных | Training | Evaluation | Комментарий |
|---|---|---|---|---|
| [Main experiment](scripts/main_exp/) | `scripts/main_exp/0-download_data.sh` | `scripts/main_exp/1-train.sh` | `scripts/main_exp/eval/*.sh` | Воспроизводит основные эксперименты статьи. |
| [NIAH](scripts/niah/) | `scripts/niah/0-gen_data.sh` | `scripts/niah/1-train.sh` | `scripts/niah/2-eval.sh` | Needle-in-a-haystack setup. |

## Viewer для self-generated данных

После скачивания или генерации данных можно посмотреть примеры через viewer:

```bash
uv run webui/self_gen_viewer.py
```

Подробнее: `webui/SELF_GEN_VIEWER.md`.

## Цитирование

```bibtex
@techreport{sakana2025doc-to-lora,
  title       = {{Doc-to-LoRA: Learning to Instantly Internalize Contexts}},
  author      = {Rujikorn Charakorn and Edoardo Cetin and Shinnosuke Uesaka and Robert Tjarko Lange},
  institution = {Sakana AI},
  year        = {2026},
  month       = {Febuary},
  note        = {Technical Report}
}
```
