# Microbrewery

Distill conversational models into any causal LM architecture on HuggingFace!

## Installation

### From PyPI

```bash
python3 -m pip install microbrewery
```

### Manual

1. Clone this repository
2. Create a new virtualenv and activate it
3. (optional) Download [PyTorch](https://pytorch.org/get-started/locally/) wheel suited to your needs
4. `pip install -r requirements.txt`

### You're all set

Done! You can now run `microbrewery --help` to see a list of available options.

## Usage

### CLI

![Bielik teaching GPT-2 Polish grammar](assets/Bielik_teaching_GPT2.png)

Distill knowledge of the Polish language from _Bielik v3.0_ into _GPT-2_ (commands tested on RTX 3070).

```bash
microbrewery distill \
    --teacher-model-path speakleash/Bielik-1.5B-v3.0-Instruct \
    --student-model-path openai-community/gpt2 \
    --dataset-path "Igorrr0/polish-qa-general" \
    --custom-system-prompt "Jesteś ekspertem od udzielania odpowiedzi, dobrze znającym język polski. Odpowiadaj krótko, konwersacyjnie, zgodnie z prawdą." \
    --completion-column-name output \
    --prompt-column-name instruction \
    --output-dir "./microbrewery-distilled-model" \
    --max-new-tokens 128 \
    --training-max-tokens 256 \
    --teacher-batch-size 32 \
    --cached-targets-path "./microbrewery-cached" \
    --learning-rate 1e-4 \
    --num-train-epochs 10 \
    --gradient-accumulation-steps 1 \
    --per-device-train-batch-size 4
```

Make sure the models are not too big and the batch size fits your VRAM.

To get a new response from the distilled model, simply run:

```bash
microbrewery infer \
    --system-prompt "Jesteś ekspertem od udzielania odpowiedzi, dobrze znającym język polski. Odpowiadaj krótko, konwersacyjnie, zgodnie z prawdą." \
    --user-prompt "Czy już znasz język polski?" \
    --model-path "./microbrewery-distilled-model/checkpoint-740"
```

![Sample response after 10 training epochs (740 steps)](assets/training_10_epochs.png)

### Library

You can also run the same functions as a Python library (e.g. in Jupyter notebooks).
Simply put `from microbrewery import distill, finetune, infer` in your code.

## Features

1. Hard target distillation
2. Easy inference with the distilled weights
3. PyTorch support
4. Caching of generated targets
5. Automatic cloning of teacher architecture's chat template
6. Q&A dataset conversion into chat format

## License

This project is licensed under the [MIT License](LICENSE.md).
