# Microbrewery

Distill conversational models into any causal LM architecture on HuggingFace!

## Installation

1. Clone this repository
2. Create a new virtualenv and activate it
3. Download [PyTorch](https://pytorch.org/get-started/locally/) for your setup (CUDA recommended)
4. `pip install -r requirements.txt`
5. Done! You can now run `python microbrewery/distiller.py --help` to see a list of available options

## Usage

![Bielik teaching GPT-2 Polish grammar](assets/Bielik_teaching-GPT2.png)

Distill knowledge of the Polish language from _Bielik v3.0_ into _GPT-2_ (commands tested on RTX 3070).

```sh
python microbrewery/distiller.py distill \
    --teacher-model speakleash/Bielik-1.5B-v3.0-Instruct-FP8-Dynamic \
    --student-model openai-community/gpt2 \
    --dataset "Igorrr0/polish-qa-general" \
    --system-prompt "Jesteś ekspertem od udzielania odpowiedzi, dobrze znającym język polski. Odpowiadaj krótko, konwersacyjnie, zgodnie z prawdą." \
    --assistant-column-name output \
    --user-column-name instruction \
    --output-dir "./microbrewery-distilled-model" \
    --max-new-tokens 128 \
    --max-length 256 \
    --inference-batch-size 32 \
    --cached-targets-path "./microbrewery-cached" \
    --learning-rate 1e-4 \
    --num-train-epochs 10 \
    --gradient-accumulation-steps 1 \
    --per-device-train-batch-size 4
```

Make sure the models are not too big and the batch size fits your VRAM.

To get a new response from the distilled model, simply run:

```sh
python microbrewery/distiller.py infer \
    --system-prompt "Jesteś ekspertem od udzielania odpowiedzi, dobrze znającym język polski. Odpowiadaj krótko, konwersacyjnie, zgodnie z prawdą." \
    --user-prompt "Czy już znasz język polski?" \
    --model-path "./microbrewery-distilled-model/checkpoint-740"
```

## Features

1. Hard target distillation
2. Caching of generated targets
3. Automatic cloning of teacher architecture's chat template
4. Q&A dataset conversion into chat format
5. Easy inference with the distilled weights

## License

This project is licensed under the [MIT License](LICENSE.md).
