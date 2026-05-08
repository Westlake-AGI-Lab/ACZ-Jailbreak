# Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment

> Findings of **ACL 2026**

ACZ-Jailbreak exposes a counter-intuitive safety failure in multimodal large language models: rendering harmful text as visually degraded images can make models more likely to follow harmful instructions. This happens even when the image remains legible and OCR accuracy is still high.

The core phenomenon is the **Attack Comfort Zone (ACZ)**. At very low resolution, models cannot read the text and the attack fails; at high resolution, safety alignment is reactivated and models usually refuse. In the middle, however, degraded but readable images can overload visual recognition, delaying or weakening safety auditing. We describe this mechanism as **Visual Cognitive Overload**.

We also study a simple mitigation, **Structured Cognitive Offloading**, which forces the model to first transcribe the image, then evaluate safety, and only then respond. This serialized workflow substantially reduces ACZ-style jailbreak behavior.

<p align="center">
  <img src="assets/paper-pipeline.png" alt="ACZ-Jailbreak overview: visually degraded harmful text can bypass safety alignment, while structured cognitive offloading restores refusal behavior." width="100%">
</p>

This repository is a lightweight release of the core data and scripts used to generate DPI-controlled text images, evaluate text/image jailbreak behavior across model providers, judge model outputs, and summarize attack success rates.

> Safety notice: this repository contains harmful prompt datasets for safety research. Do not use the data or scripts to deploy attacks against real systems.

## What Is Included

- Core image generation code in `acz_jailbreak/text2image.py`.
- OpenAI-compatible evaluation client in `acz_jailbreak/llm.py`.
- CLI scripts for image generation, model evaluation, judging, and ASR analysis.
- Text-only dataset files under `data/`.

No dataset images are committed. Regenerate images locally from the text data when needed.

<p align="center">
  <img src="assets/acz-curve.png" alt="Attack Comfort Zone curve showing image ASR rising at mid DPI while OCR remains high." width="70%">
</p>

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pdf2image` requires Poppler:

```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

## Generate Images

Render text prompts at one or more DPI settings:

```bash
python scripts/generate_images.py \
  --input data/processed/labeled_roleplay_jailbreak_results.csv \
  --text-column query \
  --id-column id \
  --dpi 15 30 45 60 90 120 150 200 300 \
  --output outputs/images \
  --limit 10
```

The paper's full DPI sweep uses `{15, 30, 45, 60, 90, 120, 150, 200, 300}`. For a quick local check, keep `--limit`; for the full image set, run the same command on a server without `--limit`.

This writes one folder per DPI, then one folder per example:

```text
outputs/images/dpi60/1/1_001.png
```

## Run Evaluation

Text-only baseline:

```bash
export OPENAI_API_KEY=...

python scripts/evaluate_text.py \
  --input data/processed/dealed_behaviors.csv \
  --text-column response \
  --id-column id \
  --provider openai \
  --model gpt-4.1-2025-04-14 \
  --output outputs/results/text.json \
  --limit 10
```

Image evaluation:

```bash
python scripts/evaluate_images.py \
  --image-root outputs/images/dpi60 \
  --provider openai \
  --model gpt-4.1-2025-04-14 \
  --output outputs/results/image_dpi60.json \
  --limit 10
```

Supported provider presets:

| Provider | Environment variable(s) | Base URL |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `siliconflow` | `SILICONFLOW_API_KEY` | `https://api.siliconflow.cn/v1` |
| `dashscope` / `bailian` | `DASHSCOPE_API_KEY` or `BAILIAN_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ark` | `ARK_API_KEY` | `https://ark.cn-beijing.volces.com/api/v3` |
| `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `bigmodel` | `BIGMODEL_API_KEY` or `ZHIPUAI_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` |
| `kimi` / `moonshot` | `KIMI_API_KEY` or `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` |
| `stepfun` | `STEP_API_KEY` or `STEPFUN_API_KEY` | `https://api.stepfun.com/v1` |
| `intern` | `INTERN_API_KEY` | `https://chat.intern-ai.org.cn/api/v1` |
| `custom` | `OPENAI_API_KEY` by default | pass `--base-url` |

Examples:

```bash
python scripts/evaluate_images.py \
  --image-root outputs/images/dpi60 \
  --provider dashscope \
  --model qwen3-vl-plus \
  --enable-thinking \
  --thinking-budget 8192 \
  --output outputs/results/qwen_image_dpi60.json
```

For any other OpenAI-compatible endpoint, use `custom` and pass `--base-url` / `--api-key-env`:

```bash
python scripts/evaluate_images.py \
  --image-root outputs/images/dpi60 \
  --provider custom \
  --model your-model \
  --base-url https://your-endpoint.example/v1 \
  --api-key-env YOUR_API_KEY_ENV \
  --output outputs/results/image_dpi60.json
```

## Judge And Analyze

The script below is a single-judge convenience path for quick analysis across providers. The paper's final reported ASR uses a three-judge protocol with human adjudication for disagreements.

Attach binary SAFE/UNSAFE scores to model outputs:

```bash
python scripts/judge_results.py \
  --input outputs/results/image_dpi60.json \
  --provider openai \
  --model gpt-4.1-2025-04-14 \
  --score-key eval_score \
  --output outputs/results/image_dpi60_judged.json
```

Summarize attack success rate:

```bash
python scripts/analyze_results.py \
  --input outputs/results/image_dpi60_judged.json \
  --score-key eval_score \
  --output outputs/summary/image_dpi60.json
```

## Acknowledgement

The text-to-image renderer follows the visual-text rendering setup of [Glyph](https://github.com/thu-coai/Glyph), using ReportLab for vector typesetting and pdf2image for rasterization.

## Citation

```bibtex
@inproceedings{song2026aczjailbreak,
  title = {Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment},
  author = {Song, Zhixue and Han, Boyan and Wang, Yiwei and Zhang, Chi},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year = {2026}
}
```
