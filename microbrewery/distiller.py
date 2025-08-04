import argparse
import logging
import os
from pathlib import Path
import datasets
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from peft import LoraConfig, TaskType
from tqdm.auto import tqdm
from trl import SFTConfig, SFTTrainer
import trl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device {DEVICE}")


def pc_to_conversational_pc(dataset, prompt_column_name, completion_column_name, custom_system_prompt):
    def to_chat_format(sample):
        if custom_system_prompt:
            system_msg = {"role": "system", "content": custom_system_prompt}
        user_msg = {"role": "user", "content": sample[prompt_column_name]}
        assistant_msg = {"role": "assistant", "content": sample[completion_column_name]}
        return {
            "prompt": [system_msg, user_msg] if custom_system_prompt else [user_msg],
            "completion": [assistant_msg]
        }
    if prompt_column_name is not None and completion_column_name is not None:
        return dataset.map(to_chat_format, remove_columns=[prompt_column_name, completion_column_name])
    else:
        raise ValueError("prompt_column_name and completion_column_name are both required fields")


def generate_hard_targets(
    teacher_model,
    teacher_tokenizer,
    dataset_path,
    batch_size=4,
    max_new_tokens=128,
    custom_system_prompt=None,
    prompt_column_name=None,
    completion_column_name=None,
):
    dataset = datasets.load_dataset(dataset_path)

    if prompt_column_name or completion_column_name:
        dataset = pc_to_conversational_pc(dataset, prompt_column_name, completion_column_name, custom_system_prompt)
    train_dataset = KeyDataset(dataset["train"], "prompt")

    assert all(m["content"] is not None
           for chat in train_dataset
           for m in chat), "Found a None content in chats"
    
    logging.debug(f"pad {teacher_tokenizer.pad_token_id}")
    logging.debug(f"eos {teacher_tokenizer.eos_token_id}")

    pipe = TextGenerationPipeline(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        device=DEVICE,
        pad_token_id=teacher_tokenizer.pad_token_id,
        eos_token_id=teacher_tokenizer.eos_token_id,
    )

    logging.info("Started pipeline")
    generated = pipe(
        train_dataset, 
        batch_size=batch_size,
        max_new_tokens=max_new_tokens, 
        do_sample=True,
    )
    logging.info("Finished pipeline")

    list_dataset = [
        {"completion": [x[0]["generated_text"][-1]], "prompt": dataset["train"][i]["prompt"]}
        for i, x in enumerate(generated)
    ]
    idx = int(len(list_dataset) * 0.8)  # 80/20 split
    targets_train = Dataset.from_list(list_dataset[:idx])
    targets_test = Dataset.from_list(list_dataset[idx : len(list_dataset)])

    return targets_train, targets_test


def train_student_model(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=512,
    lora_targets=None,
):
    if lora_targets:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=lora_targets,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        lora_config = None

    training_args = SFTConfig(
        output_dir=output_dir,
        # For prompt-completion datasets (incl. conversational PC) completion_only_loss is sufficient
        # assistant_only_loss=True,
        completion_only_loss=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        eval_strategy="steps",
        remove_unused_columns=True,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    model.train()
    trainer.train()

    return model, tokenizer


def generate_from_prompt(prompt, tokenizer, model, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(inputs, return_tensors="pt").to(DEVICE)
    out_ids = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequence = out_ids[0].tolist()
    end_of_messages_id = tokenizer.convert_tokens_to_ids("</s>")
    if end_of_messages_id in sequence:
        cut_at = sequence.index(end_of_messages_id)
        sequence = sequence[:cut_at + 1]

    return tokenizer.decode(sequence)


def finetune(args):
    student_model_path = args.student_model
    dataset_path = args.dataset
    custom_system_prompt = args.system_prompt
    chat_template_tokenizer = args.chat_template_tokenizer
    completion_column_name = args.completion_column_name
    prompt_column_name = args.prompt_column_name

    # Inference
    max_new_tokens = int(args.max_new_tokens)

    # Training
    lora_targets = args.lora_targets.split(",") if args.lora_targets is not None else None
    learning_rate = float(args.learning_rate)
    per_device_train_batch_size = int(args.per_device_train_batch_size)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    num_train_epochs = int(args.num_train_epochs)
    max_length = int(args.max_length)

    # Meta
    verbose = args.verbose
    output_dir = args.output_dir

    dataset = load_dataset(dataset_path)

    if completion_column_name or prompt_column_name:
        dataset = pc_to_conversational_pc(
            dataset, 
            prompt_column_name,
            completion_column_name,
            custom_system_prompt
        )

    model = AutoModelForCausalLM.from_pretrained(student_model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)

    if chat_template_tokenizer:
        model, tokenizer, _ = trl.clone_chat_template(
            model, tokenizer, source_tokenizer_path=chat_template_tokenizer
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if "test" not in dataset.column_names:
        # 75/25 split if not provided
        dataset = dataset["train"].train_test_split()

    targets_train = dataset["train"]
    targets_test = dataset["test"]

    sample_dataset_response = tokenizer.apply_chat_template(targets_test[0]["completion"], tokenize=False)
    sample_before_response = generate_from_prompt(
        targets_test[0]["prompt"], 
        tokenizer, 
        model,
        max_new_tokens=max_new_tokens,
    )

    # Training
    logging.info("Training student model")
    torch.cuda.empty_cache()
    model, tokenizer = train_student_model(
        model,
        tokenizer,
        train_dataset=targets_train,
        test_dataset=targets_test,
        output_dir = output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        lora_targets=lora_targets,
    )
    logging.info("Finished training")
    
    # Show sample completions
    sample_after_response = generate_from_prompt(
        targets_test[0]["prompt"],
        tokenizer,
        model,
        max_new_tokens=max_new_tokens
    )
    print("Sample response generated by teacher:")
    print(sample_dataset_response)
    print("\nSample response generated by student before training:")
    print(sample_before_response)
    print("\nSample response generated by student after training:")
    print(sample_after_response)


def distill(args):
    # General settings
    teacher_model_path = args.teacher_model
    student_model_path = args.student_model
    dataset_path = args.dataset
    custom_system_prompt = args.system_prompt

    # Teacher model
    max_new_tokens = int(args.max_new_tokens)
    inference_batch_size = int(args.inference_batch_size)
    assistant_column_name = args.assistant_column_name
    user_column_name = args.user_column_name

    # Student model
    lora_targets = args.lora_targets.split(",") if args.lora_targets is not None else None
    learning_rate = float(args.learning_rate)
    per_device_train_batch_size = int(args.per_device_train_batch_size)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    num_train_epochs = int(args.num_train_epochs)
    max_length = int(args.max_length)

    # Meta
    cached_targets_path = args.cached_targets_path
    verbose = args.verbose
    output_dir = args.output_dir
    
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


    logging.info("Starting distillation")

    # Hard target caching
    if cached_targets_path is None or not os.path.exists(cached_targets_path):
        logging.info("No cached targets found, generating teacher responses")
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(DEVICE)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

        if teacher_tokenizer.pad_token is None:
            logging.info("Pad token not found for teacher tokenizer, setting to [PAD]")
            teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            teacher_model.resize_token_embeddings(len(teacher_tokenizer))

        targets_train, targets_test = generate_hard_targets(
            teacher_model,
            teacher_tokenizer,
            dataset_path=dataset_path,
            max_new_tokens=max_new_tokens,
            batch_size=inference_batch_size,
            custom_system_prompt=custom_system_prompt,
            prompt_column_name=user_column_name,
            completion_column_name=assistant_column_name
        )
        del teacher_model, teacher_tokenizer

        if cached_targets_path is not None:
            train_path = Path(cached_targets_path) / "train.json"
            test_path = Path(cached_targets_path) / "test.json"
            targets_train.to_json(train_path)
            targets_test.to_json(test_path)
    else:
        train_path = Path(cached_targets_path) / "train.json"
        test_path = Path(cached_targets_path) / "test.json"
        logging.info(f"Responses already cached, using {train_path} and {test_path}")
        dataset = load_dataset("json", data_files={"train": str(train_path), "test": str(test_path)})
        targets_train = dataset["train"]
        targets_test = dataset["test"]

    # Student model learning
    # Initialization
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(student_model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    model, tokenizer, _ = trl.clone_chat_template(
        model, tokenizer, source_tokenizer_path=teacher_model_path
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    sample_teacher_response = tokenizer.apply_chat_template(targets_test[0]["completion"], tokenize=False)
    sample_before_response = generate_from_prompt(
        targets_test[0]["prompt"], 
        tokenizer, 
        model,
        max_new_tokens=max_new_tokens,
    )

    # Training
    logging.info("Training student model")
    model, tokenizer = train_student_model(
        model,
        tokenizer,
        train_dataset=targets_train,
        test_dataset=targets_test,
        output_dir = output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        lora_targets=lora_targets,
    )
    logging.info("Finished training")
    
    # Show sample completions
    sample_after_response = generate_from_prompt(
        targets_test[0]["prompt"],
        tokenizer,
        model,
        max_new_tokens=max_new_tokens
    )
    print("Sample response generated by teacher:")
    print(sample_teacher_response)
    print("\nSample response generated by student before training:")
    print(sample_before_response)
    print("\nSample response generated by student after training:")
    print(sample_after_response)


def infer(args):
    system_prompt = args.system_prompt
    user_prompt = args.user_prompt
    model_path = args.model_path

    if not os.path.exists(model_path):
        logging.error(f"No model found in {model_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(
        generate_from_prompt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenizer=tokenizer,
            model=model,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Microbrewery"
    )
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Available modes"
    )

    ## Fine-tuning mode##
    p_ft = subparsers.add_parser(
        "finetune", help="Fine-tune model to respond in conversation format"
    )
    p_ft.add_argument(
        "--student-model", required=True, help="Name of the student model"
    )
    p_ft.add_argument(
        "--dataset", required=True, help="Name of the dataset"
    )
    p_ft.add_argument(
        "--chat-template-tokenizer", default=None, help="If the current model is not conversational, clone this chat template"
    )
    p_ft.add_argument(
        "--system-prompt", default=None, help="System prompt text"
    )

    p_ft.add_argument(
        "--verbose", action="store_true", help="Show debug messages (flag)"
    )
    p_ft.add_argument(
        "--learning-rate", default=1e-5, help="Learning rate for SFTConfig"
    )
    p_ft.add_argument(
        "--per-device-train-batch-size",
        default=1,
        help="Train batch size per device for SFTConfig",
    )
    p_ft.add_argument(
        "--gradient-accumulation-steps",
        default=8,
        help="Gradient accumulation steps for SFTConfig",
    )
    p_ft.add_argument(
        "--num-train-epochs", default=1, help="Number of training epochs for SFTConfig"
    )
    p_ft.add_argument(
        "--max-length", default=256, help="Max length of prompt + completion in tokens (used when training)"
    )
    p_ft.add_argument(
        "--max-new-tokens",
        default=128,
        help="Max new tokens generated by model (used for before/after responses)",
    )
    p_ft.add_argument(
        "--output-dir",
        default="./microbrewery-distilled",
        help="Path to save tuned student model's weights",
    )
    p_ft.add_argument(
        "--completion-column-name",
        default=None,
        help="Name of the assistant column (optional, only for Q&A datasets)",
    )
    p_ft.add_argument(
        "--prompt-column-name",
        default=None,
        help="Name of the user column (optional; only used if --assistant-column-name is set)",
    )

    p_ft.add_argument(
        "--lora-targets",
        default=None,
        help="Uses LoRA on the target modules for training; separated with ','"
    )
    p_ft.set_defaults(func=finetune)

    ## Distillation mode ##
    p_distill = subparsers.add_parser(
        "distill", help="Distill teacher model's knowledge into student model's weights"
    )
    p_distill.add_argument(
        "--teacher-model", required=True, help="Name of the teacher model"
    )
    p_distill.add_argument(
        "--student-model", required=True, help="Name of the student model"
    )
    p_distill.add_argument(
        "--dataset", required=True, help="Name of the dataset"
    )
    p_distill.add_argument(
        "--system-prompt", required=True, help="System prompt text"
    )

    p_distill.add_argument(
        "--lora-targets",
        default=None,
        help="Uses LoRA on the target modules for training; separated with ','"
    )
    p_distill.add_argument(
        "--verbose", action="store_true", help="Show debug messages (flag)"
    )
    p_distill.add_argument(
        "--learning-rate", default=1e-5, help="Learning rate for SFTConfig"
    )
    p_distill.add_argument(
        "--per-device-train-batch-size",
        default=1,
        help="Train batch size per device for SFTConfig",
    )
    p_distill.add_argument(
        "--gradient-accumulation-steps",
        default=8,
        help="Gradient accumulation steps for SFTConfig",
    )
    p_distill.add_argument(
        "--num-train-epochs", default=1, help="Number of training epochs for SFTConfig"
    )
    p_distill.add_argument(
        "--max-length", default=256, help="Max length of prompt + completion in tokens (used when training)"
    )
    p_distill.add_argument(
        "--max-new-tokens",
        default=128,
        help="Max new tokens generated by model (used when generating, including targets by teacher model)",
    )
    p_distill.add_argument(
        "--inference-batch-size",
        default=4,
        help="Batch size when generating teacher targets",
    )
    p_distill.add_argument(
        "--cached-targets-path",
        default=None,
        help="Path of cached teacher model targets",
    )
    p_distill.add_argument(
        "--output-dir",
        default="./microbrewery-distilled",
        help="Path to save tuned student model's weights",
    )
    p_distill.add_argument(
        "--assistant-column-name",
        default=None,
        help="Name of the assistant column (optional, only for Q&A datasets)",
    )
    p_distill.add_argument(
        "--user-column-name",
        default=None,
        help="Name of the user column (optional; only used if --assistant-column-name is set)",
    )
    p_distill.set_defaults(func=distill)

    ## Inference mode ##
    p_infer = subparsers.add_parser(
        "infer", help="Generate responses using previously distilled model"
    )
    p_infer.add_argument(
        "--system-prompt", 
        required=True, 
        help="System prompt text"
    )
    p_infer.add_argument(
        "--user-prompt", 
        required=True, 
        help="User prompt text"
    )
    p_infer.add_argument(
        "--model-path", 
        required=True, 
        help="Path to a folder containing distilled model's weights"
    )
    p_infer.set_defaults(func=infer)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
    main()
