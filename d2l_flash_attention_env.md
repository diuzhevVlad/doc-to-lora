# Окружение D2L с Flash-Attention

## Цель

Для проекта создана отдельная `conda`-среда `d2l-flash` с `flash-attn`. Она нужна, чтобы запускать D2L evaluation изолированно от системного Python и проверить, ускоряет ли Flash-Attention полный SQuAD-прогон.

Итог: окружение работает, `flash-attn` импортируется, но на текущем evaluation setup ускорение не стало главным фактором. Основное время уходит в autoregressive generation и D2L-specific операции.

## Машина

| Компонент | Значение |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU |
| VRAM | 16 GB |
| Driver | 580.126.09 |
| PyTorch CUDA runtime | 12.4 |
| Python | 3.10 |
| Conda env | `d2l-flash` |

`nvcc` не требуется, потому что используется готовый wheel `flash-attn`.

## Создание среды

```bash
conda create -y -n d2l-flash python=3.10 pip
```

Базовые build/runtime зависимости:

```bash
conda run -n d2l-flash python -m pip install --upgrade \
  pip setuptools wheel packaging ninja
```

PyTorch с CUDA 12.4:

```bash
conda run -n d2l-flash python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

`flash-attn` wheel для `torch==2.6.0`, CUDA 12 и Python 3.10:

```bash
conda run -n d2l-flash python -m pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Остальные зависимости evaluation:

```bash
conda run -n d2l-flash python -m pip install \
  transformers==4.51.3 accelerate==1.6.0 datasets==3.6.0 peft \
  tensorboardX rouge-score jaxtyping opt-einsum llmlingua bitsandbytes \
  pandas hf_transfer "huggingface-hub[hf-transfer]>=0.32.0" \
  tokenizers==0.21.0 wandb inflect torchmetrics
```

## Проверка установки

```bash
conda run -n d2l-flash python -c \
  "import torch, flash_attn; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(flash_attn.__version__)"
```

Ожидаемый результат:

```text
2.6.0+cu124 12.4 True
2.7.4.post1
```

Проверка импортов проекта:

```bash
PYTHONPATH=src conda run -n d2l-flash python -c \
  "import run_eval; import ctx_to_lora.eval_utils; print('imports ok')"
```

Ожидаемый результат:

```text
imports ok
```

## Команда D2L-прогона

Pilot на 100 примерах:

```bash
SECONDS=0
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
TRANSFORMERS_NO_TF=1 \
HF_HUB_ENABLE_HF_TRANSFER=0 \
WANDB_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n d2l-flash python run_eval.py \
  --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  --datasets \
    d2l_hypothesis/squad/content_adapter \
    d2l_hypothesis/squad/content_wrong \
    d2l_hypothesis/squad/instruction_adapter \
    d2l_hypothesis/squad/instruction_wrong \
  --split validation \
  --eval_batch_size_gen 1 \
  --max_val_samples_per_ds 100 \
  --max_new_tokens 64
echo "ELAPSED_SECONDS=$SECONDS"
```

Полный SQuAD D2L-прогон запускался аналогично, но без `--max_val_samples_per_ds 100`.

## Pilot benchmark

Wall-clock runtime для 4 D2L-условий по 100 примеров: `914` секунд, то есть `15м14с`.

Оценка по условиям:

| condition | samples | runtime на 100 | samples/sec | оценка на полный SQuAD |
|---|---:|---:|---:|---:|
| `content_adapter` | 100 | 4м36с | 0.363 | ~8ч06м |
| `content_wrong` | 100 | 5м31с | 0.302 | ~9ч44м |
| `instruction_adapter` | 100 | 2м14с | 0.748 | ~3ч55м |
| `instruction_wrong` | 100 | 2м10с | 0.771 | ~3ч49м |

Суммарная оценка для полного SQuAD была `26-28 часов`.

## Фактический полный D2L runtime

Полный D2L-прогон завершился примерно в ожидаемом диапазоне.

| condition | samples | runtime | samples/sec |
|---|---:|---:|---:|
| `content_adapter` | 10,570 | 8:35:40 | 0.342 |
| `content_wrong` | 10,570 | 9:23:29 | 0.313 |
| `instruction_adapter` | 10,570 | 3:32:41 | 0.828 |
| `instruction_wrong` | 10,570 | 3:36:22 | 0.814 |

Сумма measured evaluation runtime: около `25ч08м`. Полный wall-clock может быть немного больше из-за загрузки модели, tokenization, записи generated outputs и подсчета метрик.

## Base-model controls runtime

Controls запущены отдельно в `tmux` session:

```text
d2l_squad_controls
```

Готовые runtime:

| control | samples | runtime | samples/sec |
|---|---:|---:|---:|
| `prompt_only_no_context` | 10,570 | 1:58:43 | 1.484 |
| `content_adapter_no_context` | 10,570 | 6:43:48 | 0.436 |
| `instruction_adapter_no_context` | in progress | in progress | in progress |

## Почему Flash-Attention не дал большого ускорения

В этом setup `eval_batch_size_gen=1`, а generation autoregressive. Поэтому ускорение attention не доминирует над всем runtime.

Кроме attention, время тратится на:

- D2L `generate_weights`;
- применение LoRA weights;
- обработку длинных prompts;
- генерацию токенов до `max_new_tokens=64`;
- запись per-sample outputs и подсчет метрик.

Flash-Attention остается полезным как совместимое и потенциально более memory-efficient окружение, но ожидать кратного ускорения здесь не стоит.

## Изменения в репозитории

Временные fallback-правки были убраны. В исходном коде оставлен только memory cleanup в:

```text
src/ctx_to_lora/modeling/hypernet.py
```

Смысл cleanup: перед повторной инициализацией checkpoint удалить старые `hypernet` и `ctx_encoder`, затем очистить CUDA cache. На 16 GB GPU это предотвращает CUDA OOM до начала evaluation.

## Как проверить текущий прогон

```bash
tmux attach -t d2l_squad_controls
```

Лог:

```text
logs/d2l_squad_controls/full_squad_controls_20260509-230145.log
```

CSV controls:

```text
eval_results/Qwen/Qwen3-4B-Instruct-2507/20260509-230150_f4e0af7f/evaluation_results_generation_no_context.csv
```

