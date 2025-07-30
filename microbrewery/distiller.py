import argparse
import logging
import os
import datasets
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from peft import LoraConfig, TaskType
from tqdm.auto import tqdm
from trl import SFTConfig, SFTTrainer# New arguments for custom cache paths
import trl

TEACHER_MODEL = "speakleash/Bielik-1.5B-v3.0-Instruct"
STUDENT_MODEL = "sdadas/polish-gpt2-small"
DATASET = "Igorrr0/polish-qa-general"
SYSTEM_PROMPT = "Odpowiadaj krótko i konwersacyjnie :)"
USE_LORA = False

DEFAULT_TRAIN_PATH = "./train.json"
DEFAULT_TEST_PATH = "./test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device {DEVICE}")

def generate_hard_targets(teacher_model_path, dataset_path, train_path, test_path, user_prompt_column=None, assistant_output_column=None):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(DEVICE)

    dataset = datasets.load_dataset(dataset_path)

    def to_chat_format(sample):
        system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": sample["instruction"]}
        assistant_msg = {"role": "assistant", "content": sample["output"]}
        return {"prompt": [system_prompt, user_msg], "completion": [system_prompt, user_msg, assistant_msg]}

    if user_prompt_column is not None and assistant_output_column is not None:
        dataset = dataset.map(to_chat_format).remove_columns(["instruction", "input", "output"])
    elif user_prompt_column is None and assistant_output_column is None:
        pass
    else:
        raise ValueError("Both user_prompt_column and assistant_output_column need to be set or None at the same time")

    pipe = TextGenerationPipeline(
        model=model, 
        tokenizer=tokenizer,
        batch_size=1,
        device=DEVICE,
    )

    train_dataset = KeyDataset(dataset["train"], "prompt")

    print("Generating responses")
    generated = pipe(train_dataset, batch_size=64, max_new_tokens=128, do_sample=True)
    print("Done")

    print(generated[0])
    list_dataset = [{"completion": x[0]["generated_text"], "prompt": dataset["train"][i]["prompt"]} for i, x in enumerate(generated)]
    idx = int(len(list_dataset) * 0.8)  # 80/20 split
    cached_train = Dataset.from_list(list_dataset[:idx])
    cached_train.to_json(train_path)
    cached_test = Dataset.from_list(list_dataset[idx:len(list_dataset)])
    cached_test.to_json(test_path)


def generate_from_prompt(prompt, tokenizer, model):
    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    out_ids = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        temperature=0.0
    )

    sequence = out_ids[0].tolist()
    print(tokenizer.decode(tokenizer.eos_token_id))
    if tokenizer.eos_token_id in sequence:
        cut_at = sequence.index(tokenizer.eos_token_id)
        sequence = sequence[:cut_at+1]

    return tokenizer.decode(sequence)


def train_student_model(model_path, teacher_tokenizer_path, train_path, test_path):
    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    logging.debug(train_dataset)
    logging.debug(train_dataset[0])
    logging.debug(test_dataset)
    logging.debug(test_dataset[0])

    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_fc", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM 
    )

    trl.clone_chat_template(model, tokenizer, source_tokenizer_path=teacher_tokenizer_path)

    tokenizer.eos_token_id = tokenizer.encode("</s>")[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    training_args = SFTConfig(
    output_dir="student_model",
    max_length=256,
    assistant_only_loss=True,
    per_device_train_batch_size=8, 
    gradient_accumulation_steps=1,
    remove_unused_columns=False,
    learning_rate=1e-5,
    eval_strategy="steps",
    num_train_epochs=3

)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=lora_config if USE_LORA else None
    )

    trainer.train()

    return model, tokenizer


def distill(args):
    TEACHER_MODEL          = args.teacher_model
    STUDENT_MODEL          = args.student_model
    DATASET                = args.dataset
    SYSTEM_PROMPT          = args.system_prompt
    USE_LORA               = args.use_lora
    VERBOSE                = args.verbose
    train_path             = args.train_dataset_path
    test_path              = args.test_dataset_path
    assistant_column_name  = args.assistant_column_name
    user_column_name       = args.user_column_name

    if VERBOSE:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    print("Starting generation...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        generate_hard_targets(teacher_model_path=TEACHER_MODEL, dataset_path=DATASET, user_prompt_column=user_column_name, assistant_output_column=assistant_column_name, train_path=train_path, test_path=test_path)
    else:
        print(f"responses already cached, using {train_path} and {test_path}")
    model, tokenizer = train_student_model(model_path=STUDENT_MODEL, teacher_tokenizer_path=TEACHER_MODEL, train_path=train_path, test_path=test_path)
    model.save_pretrained("./microbrewery-distilled")
    tokenizer.save_pretrained("./microbrewery-distilled")


def main():
    parser = argparse.ArgumentParser(description="Run Microbrewery (modes: distill, gen)")
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Available modes"
    )

    p_distill = subparsers.add_parser("distill", help="Distill teacher model's knowledge into student model's weights")
    p_distill.add_argument("--teacher-model",    required=True, help="Name of the teacher model")
    p_distill.add_argument("--student-model",    required=True, help="Name of the student model")
    p_distill.add_argument("--dataset",          required=True, help="Name of the dataset")
    p_distill.add_argument("--system-prompt",    required=True, help="System prompt text")
    p_distill.add_argument("--use-lora",         action="store_true", help="Enable LoRA (flag)")
    p_distill.add_argument("--verbose",          action="store_true", help="Show debug messages (flag)")

    p_distill.add_argument(
        "--train-dataset-path",
        default=DEFAULT_TRAIN_PATH,
        help="Path to save/load cached train dataset JSON"
    )
    p_distill.add_argument(
        "--test-dataset-path",
        default=DEFAULT_TEST_PATH,
        help="Path to save/load cached test dataset JSON"
    )

    p_distill.add_argument(
        "--assistant-column-name",
        default=None,
        help="Name of the assistant column (optional, only for Q&A datasets)"
    )
    p_distill.add_argument(
        "--user-column-name",
        default=None,
        help="Name of the user column (optional; only used if --assistant-column-name is set)"
    )
    p_distill.set_defaults(func=distill)

    args = parser.parse_args()

    # sanity check: user-column only makes sense if assistant-column was provided
    if args.user_column_name and not args.assistant_column_name:
        p_distill.error("--user-column-name requires --assistant-column-name to be set")

    args.func(args)


if __name__ == "__main__":
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
    main()
