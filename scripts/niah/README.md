# Эксперимент NIAH

Эта папка содержит needle-in-a-haystack experiment для D2L.

Запускать скрипты нужно по порядку. Генерацию данных достаточно выполнить один раз.

```bash
uv run bash scripts/niah/0-gen_data.sh
uv run bash scripts/niah/1-train.sh
uv run bash scripts/niah/2-eval.sh
```
