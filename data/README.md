# Data

This directory contains text-only data for the lightweight ACZ-Jailbreak image generation and evaluation scripts. It intentionally excludes generated image datasets and image archives.

## raw

- `advbench_harmful_behaviors.csv`: AdvBench-style harmful goals and targets.
- `harmbench_behaviors_text_all.csv`: HarmBench behavior metadata.

These raw files are source references for the released data. The paper also uses the Liu et al. jailbreak dataset during curation; it is reflected in the processed prompt set rather than released here as a separate raw file.

## processed

- `labeled_roleplay_jailbreak_results.csv`: 770 final text records used by the release scripts to generate ACZ image inputs.
- `dealed_behaviors.csv`: 120 text prompt/response pairs for quick text-only baseline evaluation.

Use the text records here with `scripts/generate_images.py` to regenerate ACZ image inputs.
