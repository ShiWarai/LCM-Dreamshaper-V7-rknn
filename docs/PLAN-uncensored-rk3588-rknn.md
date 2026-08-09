# План: Uncensored SD → RKNN на RK3588 NPU

Документ для машины с **rknn-toolkit2** (конвертер).  
Цель: получить `.rknn` модели, которые крутятся на **NPU RK3588**, с чекпоинтом менее цензурным, чем базовый SD 1.5 / Dreamshaper LCM.

**LCM не обязателен.** Главное — рабочий inference на NPU.

---

## 0. Краткий выбор стратегии

| Вариант | Скорость | Качество | Сложность | Рекомендация |
|---------|----------|----------|-----------|--------------|
| **A. SD 1.5 + Euler** (без LCM) | ~3–5 мин / картинка | выше | проще | **основной путь** |
| **B. SD 1.5 + LCM / Hyper** | ~30–90 с | ниже / «быстрее» | сложнее (нужен `timestep_cond` или другой runtime) | опционально |

Для старта берите **вариант A**. Вариант B — отдельный проход после того, как A уже работает на плате.

Готовый runtime под Euler на RK3588:  
[Mojo24x7/SD1.5_rknn_3588_euler](https://github.com/Mojo24x7/SD1.5_rknn_3588_euler)

Готовый runtime под LCM (Dreamshaper):  
[darkautism/LCM-Dreamshaper-V7-rs](https://github.com/darkautism/LCM-Dreamshaper-V7-rs)  
*(другой UNet-контракт — не смешивать с Euler-моделями без адаптации кода)*

---

## 1. Требования к машине конвертера

- **ОС:** Linux x86_64 (рекомендуется)
- **Python:** 3.10 или **3.11**
- **rknn-toolkit2:** **ровно 2.3.2** (должен совпадать с `librknnrt.so` на плате)
- **RAM:** ≥ 32 GB (UNet тяжёлый; 64 GB комфортнее)
- **Диск:** ≥ 40 GB свободно
- **GPU:** не обязателен для RKNN-конверта (можно CPU), но ускоряет экспорт ONNX из PyTorch

### Установка toolkit

```bash
# Python 3.11
uv venv /tmp/rknn-venv --python 3.11
source /tmp/rknn-venv/bin/activate

pip install setuptools==75.0.0
pip install rknn-toolkit2==2.3.2

# для экспорта ONNX (если будете экспортировать сами)
pip install "torch" "diffusers" "transformers" "accelerate" "safetensors" "optimum[onnxruntime]" "onnx" "protobuf"
```

Проверка версии:

```bash
python -c "from rknn.api import RKNN; print('OK')"
```

На **плате** должно быть:

```text
librknnrt.so 2.3.2
драйвер RKNPU ( /dev/dri/renderD* )
пользователь в группе render (или root)
```

Скачать runtime:  
https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

---

## 2. Выбор исходной модели (uncensored / permissive)

Берите **SD 1.5 architecture** (один CLIP, latent 64×64 для 512×512). SDXL / Flux на NPU RK3588 почти нереально без отдельного проекта.

### Рекомендуемые источники

| Приоритет | Модель | Формат | Комментарий |
|-----------|--------|--------|-------------|
| 1 | [TheyCallMeHex/Deliberate-V3-ONNX](https://huggingface.co/TheyCallMeHex/Deliberate-V3-ONNX) | **уже ONNX** | Меньше работы: сразу → RKNN |
| 2 | [Heliosoph/realistic-vision-cfg-onnx](https://huggingface.co/Heliosoph/realistic-vision-cfg-onnx) | уже ONNX | Realistic Vision V6, качество |
| 3 | [Heliosoph/realistic-vision-hyper-onnx](https://huggingface.co/Heliosoph/realistic-vision-hyper-onnx) | уже ONNX | 4 шага (Hyper), permissive |
| 4 | PyTorch с Civitai / HF (Deliberate, Realistic Vision, и т.п.) | safetensors | Нужен экспорт в ONNX самому |

**Предпочтительный путь:** взять **готовый ONNX** (Deliberate-V3 или Realistic Vision) → конвертировать в RKNN.  
Не тратьте время на «uncensored LCM ONNX» — готовых почти нет.

---

## 3. Вариант A (рекомендуется): ONNX → RKNN без LCM

### 3.1. Скачать ONNX

```bash
mkdir -p ~/sd-rknn-work && cd ~/sd-rknn-work

# Пример: Deliberate V3
huggingface-cli download TheyCallMeHex/Deliberate-V3-ONNX \
  --local-dir ./deliberate-v3-onnx

# или Realistic Vision CFG:
# huggingface-cli download Heliosoph/realistic-vision-cfg-onnx --local-dir ./rv6-onnx
```

Ожидаемая структура:

```text
model/
  text_encoder/model.onnx
  unet/model.onnx (+ model.onnx_data если есть)
  vae_decoder/model.onnx
  vae_encoder/model.onnx   # опционально (для img2img)
  tokenizer/
  scheduler/
  model_index.json
```

### 3.2. Скрипт конвертации в RKNN

Сохраните как `convert-onnx-to-rknn.py` (адаптация community-скрипта под SD 1.5 / 512×512):

```python
#!/usr/bin/env python3
"""Convert SD1.5 ONNX components to RKNN for RK3588 (toolkit 2.3.2)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rknn.api import RKNN

LATENT_FACTOR = 8


def convert_component(
    onnx_path: Path,
    out_path: Path,
    inputs: List[str] | None,
    input_size_list: List[List[int]] | None,
    target_platform: str = "rk3588",
) -> None:
    print(f"\n=== Converting {onnx_path} → {out_path} ===")
    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform=target_platform,
        float_dtype="float16",
        optimization_level=3,
        # single_core_mode=True  # раскомментируйте, если build падает по памяти
    )

    ret = rknn.load_onnx(
        model=str(onnx_path),
        inputs=inputs,
        input_size_list=input_size_list,
    )
    if ret != 0:
        raise SystemExit(f"load_onnx failed: {ret}")

    # FP16 без INT8 — стабильнее качество для diffusion
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(out_path))
    if ret != 0:
        raise SystemExit(f"export_rknn failed: {ret}")

    rknn.release()
    print(f"OK: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model-dir", required=True, type=Path)
    p.add_argument("-o", "--out-dir", required=True, type=Path)
    p.add_argument("-r", "--resolution", default="512x512", help="WxH, e.g. 512x512")
    p.add_argument(
        "-c",
        "--components",
        default="text_encoder,unet,vae_decoder",
        help="comma-separated",
    )
    p.add_argument("--target", default="rk3588")
    args = p.parse_args()

    w, h = map(int, args.resolution.lower().split("x"))
    lh, lw = h // LATENT_FACTOR, w // LATENT_FACTOR  # 64x64 for 512

    specs = {
        "text_encoder": {
            "inputs": ["input_ids"],
            "input_size_list": [[1, 77]],
        },
        "unet": {
            # Классический SD1.5 UNet (НЕ LCM): sample, timestep, encoder_hidden_states
            "inputs": ["sample", "timestep", "encoder_hidden_states"],
            "input_size_list": [
                [1, 4, lh, lw],
                [1],
                [1, 77, 768],
            ],
        },
        "vae_decoder": {
            "inputs": ["latent_sample"],
            "input_size_list": [[1, 4, lh, lw]],
        },
        "vae_encoder": {
            "inputs": ["sample"],
            "input_size_list": [[1, 3, h, w]],
        },
    }

    for name in [c.strip() for c in args.components.split(",")]:
        if name not in specs:
            raise SystemExit(f"Unknown component: {name}")
        onnx = args.model_dir / name / "model.onnx"
        if not onnx.is_file():
            raise SystemExit(f"Missing: {onnx}")
        out = args.out_dir / name / "model.rknn"
        convert_component(
            onnx,
            out,
            inputs=specs[name]["inputs"],
            input_size_list=specs[name]["input_size_list"],
            target_platform=args.target,
        )


if __name__ == "__main__":
    main()
```

### 3.3. Запуск конвертации

```bash
source /tmp/rknn-venv/bin/activate
cd ~/sd-rknn-work

python convert-onnx-to-rknn.py \
  -m ./deliberate-v3-onnx \
  -o ./deliberate-v3-rknn-512 \
  -r 512x512 \
  -c "text_encoder,unet,vae_decoder"

# опционально для img2img:
# -c "text_encoder,unet,vae_decoder,vae_encoder"
```

**Ожидаемое время (x86, CPU):** UNet — десятки минут–часы; text_encoder и VAE — быстрее.  
**Размер:** UNet `.rknn` порядка 1.5–2 GB (FP16).

### 3.4. Если `load_onnx` / `build` падает

1. Убедитесь, что рядом с `unet/model.onnx` лежит `model.onnx_data` (если был в архиве).
2. Имена входов могут отличаться — посмотрите через Netron / onnx:

   ```bash
   python - <<'EOF'
   import onnx
   m = onnx.load("deliberate-v3-onnx/unet/model.onnx")
   for i in m.graph.input:
       print(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
   EOF
   ```

   Подставьте реальные имена в `inputs=` и размеры в `input_size_list`.

3. Нехватка RAM: закройте браузер, поставьте swap, или `single_core_mode=True` в `rknn.config`.
4. Не используйте INT8 (`do_quantization=True`) для первого прогона — часто ломает качество diffusion.

### 3.5. Артефакты для платы

Скопируйте на RK3588:

```text
deliberate-v3-rknn-512/
  text_encoder/model.rknn
  unet/model.rknn
  vae_decoder/model.rknn
  # + tokenizer.json / scheduler_config.json из исходного ONNX-репо
```

И положите рядом `librknnrt.so` **2.3.2** (тот же major.minor.patch, что toolkit).

---

## 4. Запуск на плате (вариант A)

### Вариант A1 — готовый Euler runtime

1. Клонировать [Mojo24x7/SD1.5_rknn_3588_euler](https://github.com/Mojo24x7/SD1.5_rknn_3588_euler)
2. Подменить их `models/.../*.rknn` своими из `deliberate-v3-rknn-512/`
3. Следовать README репо (txt2img / WebUI)

### Вариант A2 — свой минимальный Python (rknn-toolkit-lite2)

Псевдопайплайн:

1. Tokenizer (CLIP) на CPU → `input_ids` [1, 77]
2. `text_encoder.rknn` → embeddings [1, 77, 768]
3. Цикл Euler / DDIM: на каждом шаге `unet.rknn(sample, timestep, emb)`  
   *(между шагами держать в NPU только UNet; TE и VAE выгружать — на RK3588 SRAM мало)*
4. `vae_decoder.rknn` → RGB 512×512

**Критично:** на RK3588 часто **нельзя держать TE + UNet + VAE одновременно** в NPU.  
Порядок: загрузить → прогнать → `rknn_release` / destroy → следующая модель.

---

## 5. Вариант B (опционально): LCM / Hyper

Делайте **только после** успешного A.

### 5.1. Hyper-SD ONNX (уже есть)

[Heliosoph/realistic-vision-hyper-onnx](https://huggingface.co/Heliosoph/realistic-vision-hyper-onnx) — RV6 + Hyper 4-step.  
Конвертация аналогична п.3, но:

- scheduler / число шагов другие (CFG≈1, ~4 шага);
- runtime должен уметь этот scheduler (не классический LCM Dreamshaper).

### 5.2. Свой LCM (сложнее)

1. Взять uncensored SD 1.5 (PyTorch).
2. Слить LoRA: `latent-consistency/lcm-lora-sdv1-5` в UNet (`fuse_lora`).
3. Экспорт:

   ```bash
   optimum-cli export onnx --model ./merged_lcm_model ./lcm_uncensored_onnx/
   ```

4. У LCM-UNet обычно **4 входа**, включая `timestep_cond` [1, 256].  
   В `convert-onnx-to-rknn.py` для unet:

   ```python
   "inputs": ["sample", "timestep", "encoder_hidden_states", "timestep_cond"],
   "input_size_list": [[1,4,64,64], [1], [1,77,768], [1,256]],
   ```

5. Inference — только LCM-совместимый код (например LCM-Dreamshaper-V7-rs), **не** Euler Mojo.

---

## 6. Если ONNX нет — экспорт из PyTorch

На машине с GPU (или мощным CPU):

```bash
# Пример: diffusers-модель или конвертированный ckpt → diffusers
optimum-cli export onnx \
  --model SG161222/Realistic_Vision_V6.0_B1_noVAE \
  --task stable-diffusion \
  ./rv6-onnx/
```

Или скрипт из diffusers:

```bash
# https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
python convert_stable_diffusion_checkpoint_to_onnx.py \
  --model_path /path/to/diffusers_model \
  --output_path ./my-onnx
```

Потом — раздел 3 (ONNX → RKNN).

Для VAE лучше подставить `stabilityai/sd-vae-ft-mse`, если чекпоинт без своего VAE.

---

## 7. Чеклист перед копированием на плату

- [ ] Toolkit на PC = **2.3.2**
- [ ] На плате `librknnrt.so` = **2.3.2** (строки version совпадают)
- [ ] Есть `text_encoder/model.rknn`, `unet/model.rknn`, `vae_decoder/model.rknn`
- [ ] Разрешение зафиксировано (например только **512×512**) — dynamic shape на RKNN хрупкий
- [ ] `do_quantization=False` (первый рабочий билд)
- [ ] Runtime на плате понимает **тип** модели (Euler vs LCM)
- [ ] Поэтапная загрузка моделей (не все три сразу в NPU)
- [ ] Пользователь в группе `render`

Проверка runtime на плате:

```bash
strings /path/to/librknnrt.so | grep -E '^[0-9]+\.[0-9]+\.[0-9]+' | head
# ожидание: 2.3.2 ...
```

---

## 8. Типичные ошибки

| Симптом | Причина | Что делать |
|---------|---------|------------|
| SIGSEGV при `rknn_init` | toolkit ≠ runtime (2.3.0 vs 2.3.2) | Пересобрать RKNN тем же 2.3.2 |
| `failed to malloc npu memory` | Все модели в NPU сразу | Выгружать TE перед UNet, UNet перед VAE |
| Картинка «каша» / серая | INT8 / bad VAE / wrong layout NHWC/NCHW | FP16; проверить layout в runtime |
| `load_onnx` fail | Нет `model.onnx_data` / неверные имена входов | Netron + поправить `inputs` |
| UNet не подходит к dreamshaper-cli | Это Euler-UNet без `timestep_cond` | Использовать Mojo Euler runtime |

---

## 9. Рекомендуемый порядок работ (коротко)

1. На PC с toolkit **2.3.2** скачать **Deliberate-V3-ONNX** (или RV6 ONNX).
2. Сконвертировать `text_encoder`, `unet`, `vae_decoder` → `.rknn` (512×512, FP16).
3. Скопировать на RK3588 + `librknnrt 2.3.2`.
4. Запустить через **Mojo24x7 Euler** (подмена моделей) или свой Python pipeline.
5. Убедиться, что txt2img стабилен.
6. *(Опционально)* повторить с Hyper / своим LCM под быстрый режим.

---

## 10. Что отправить обратно / критерии успеха

Критерий «готово»:

1. На плате команда txt2img отрабатывает без crash.
2. Получается узнаваемая картинка 512×512 по промпту.
3. NPU грузится (`rknputop` / load > 0 во время UNet).

Полезные файлы для отладки с конвертера:

- лог `rknn.build` (verbose)
- вывод имён/shape входов ONNX
- `md5sum` всех `.rknn`
- точная строка версии `librknnrt.so` с платы

---

## Ссылки

- RKNN toolkit2: https://github.com/airockchip/rknn-toolkit2  
- Deliberate ONNX: https://huggingface.co/TheyCallMeHex/Deliberate-V3-ONNX  
- Realistic Vision ONNX: https://huggingface.co/Heliosoph/realistic-vision-cfg-onnx  
- Euler runtime RK3588: https://github.com/Mojo24x7/SD1.5_rknn_3588_euler  
- LCM Dreamshaper (для сравнения / не смешивать UNet): https://github.com/darkautism/LCM-Dreamshaper-V7-rs  
- Optimum ONNX export: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model  

---

*Версия плана: 2026-08-05. Целевая платформа: Rockchip RK3588 + rknn-toolkit2 / librknnrt 2.3.2.*
