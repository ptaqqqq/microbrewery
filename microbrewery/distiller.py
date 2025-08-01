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


DEFAULT_TRAIN_PATH = "./train.json"
DEFAULT_TEST_PATH = "./test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device {DEVICE}")


def generate_hard_targets(
    teacher_model_path, 
    dataset_path, 
    train_path,
    test_path,
    custom_system_prompt=None, 
    prompt_column_name=None, 
    completion_column_name=None
):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(DEVICE)

    dataset = datasets.load_dataset(dataset_path)

    def to_chat_format(sample):
        if custom_system_prompt:
            system_msg = {"role": "system", "content": custom_system_prompt}
        user_msg = {"role": "user", "content": sample[prompt_column_name]}
        assistant_msg = {"role": "assistant", "content": sample[completion_column_name]}
        return {
            "prompt": [system_msg, user_msg] if custom_system_prompt else [user_msg], 
            "completion": [system_msg, user_msg, assistant_msg] if custom_system_prompt else [user_msg, assistant_msg]
        }

    if prompt_column_name is not None and completion_column_name is not None:
        dataset = dataset.map(to_chat_format).remove_columns([prompt_column_name, completion_column_name])
    elif prompt_column_name is None and completion_column_name is None:
        pass
    else:
        raise ValueError("Both user_prompt_column and assistant_output_column need to be set or None at the same time")

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    pipe = TextGenerationPipeline(
        model=model, 
        tokenizer=tokenizer,
        batch_size=1,
        device=DEVICE,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
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


def train_student_model(
    model_path,
    teacher_tokenizer_path, 
    train_path, 
    test_path, 
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_length=512,
    use_lora=False
):
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

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    model.resize_token_embeddings(len(tokenizer))
    print(f"pad token {tokenizer.pad_token}")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    training_args = SFTConfig(
        output_dir="student_model",
        assistant_only_loss=True,
        completion_only_loss=True,

        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        eval_strategy="steps",

        remove_unused_columns=False,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=lora_config if use_lora else None
    )

    trainer.train()

    return model, tokenizer


def generate_from_prompt(prompt, tokenizer, model):
    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(inputs, return_tensors="pt").to(DEVICE)
    out_ids = model.generate(
        **input_ids,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    sequence = out_ids[0].tolist()
    if tokenizer.convert_tokens_to_ids("</s>") in sequence:
        cut_at = sequence.index(tokenizer.eos_token_id)
        sequence = sequence[:cut_at+1]

    return tokenizer.decode(sequence)


def distill(args):
    TEACHER_MODEL                = args.teacher_model
    STUDENT_MODEL                = args.student_model
    DATASET                      = args.dataset
    custom_system_prompt         = args.system_prompt
    use_lora                     = args.use_lora
    verbose                      = args.verbose
    learning_rate                = float(args.learning_rate)
    per_device_train_batch_size  = int(args.per_device_train_batch_size)
    gradient_accumulation_steps  = int(args.gradient_accumulation_steps)
    num_train_epochs             = int(args.num_train_epochs)
    max_length                   = int(args.max_length)
    train_path                   = args.train_dataset_path
    test_path                    = args.test_dataset_path
    assistant_column_name        = args.assistant_column_name
    user_column_name             = args.user_column_name

    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    print("Starting generation...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        generate_hard_targets(
            teacher_model_path=TEACHER_MODEL, 
            dataset_path=DATASET, 
            custom_system_prompt=custom_system_prompt,
            prompt_column_name=user_column_name,
            completion_column_name=assistant_column_name, 
            train_path=train_path, 
            test_path=test_path
        )
    else:
        print(f"responses already cached, using {train_path} and {test_path}")
    model, tokenizer = train_student_model(
        model_path=STUDENT_MODEL,
        teacher_tokenizer_path=TEACHER_MODEL, 
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        train_path=train_path, 
        test_path=test_path,
        use_lora=use_lora
    )
    model.save_pretrained("./microbrewery-distilled")
    tokenizer.save_pretrained("./microbrewery-distilled")


def infer(args):
    system_prompt = args.system_prompt
    user_prompt = args.user_prompt

    model = AutoModelForCausalLM.from_pretrained("./microbrewery-distilled").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("./microbrewery-distilled")

    print(generate_from_prompt([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], tokenizer=tokenizer, model=model))


def main():
    parser = argparse.ArgumentParser(description="Run Microbrewery (modes: distill, gen)")
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Available modes"
    )

    ## Distillation mode ##
    p_distill = subparsers.add_parser("distill", help="Distill teacher model's knowledge into student model's weights")
    p_distill.add_argument("--teacher-model",    required=True, help="Name of the teacher model")
    p_distill.add_argument("--student-model",    required=True, help="Name of the student model")
    p_distill.add_argument("--dataset",          required=True, help="Name of the dataset")
    p_distill.add_argument("--system-prompt",    required=True, help="System prompt text")
    p_distill.add_argument("--use-lora",         action="store_true", help="Enable LoRA (flag)")
    p_distill.add_argument("--verbose",          action="store_true", help="Show debug messages (flag)")
    p_distill.add_argument(
        "--learning-rate",
        default=1e-5,
        help="Learning rate for SFTConfig"
    )
    p_distill.add_argument(
        "--per-device-train-batch-size",
        default=1,
        help="Train batch size per device for SFTConfig"
    )
    p_distill.add_argument(
        "--gradient-accumulation-steps",
        default=8,
        help="Gradient accumulation steps for SFTConfig"
    )
    p_distill.add_argument(
        "--num-train-epochs",
        default=1,
        help="Number of training epochs for SFTConfig"
    )
    p_distill.add_argument(
        "--max-length",
        default=256,
        help="Max length of prompt + completion in tokens"
    )
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

    ## Inference mode ##
    p_infer = subparsers.add_parser("infer", help="Generate responses using previously distilled model")
    p_infer.set_defaults(func=infer)

    p_infer.add_argument("--system-prompt", required=True, help="System prompt text")
    p_infer.add_argument("--user-prompt", required=True, help="User prompt text")

    args = parser.parse_args()

    # sanity check: user-column only makes sense if assistant-column was provided
    # if args.user_column_name and not args.assistant_column_name:
        # p_distill.error("--user-column-name requires --assistant-column-name to be set")

    args.func(args)


if __name__ == "__main__":
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
    main()
