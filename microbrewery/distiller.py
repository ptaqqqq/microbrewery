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
from trl import SFTConfig, SFTTrainer
import trl


DEFAULT_TRAIN_PATH = "./train.json"
DEFAULT_TEST_PATH = "./test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device {DEVICE}")


def generate_hard_targets(
    teacher_model,
    teacher_tokenizer,
    dataset_path,
    train_path,
    test_path,
    batch_size=4,
    max_new_tokens=128,
    custom_system_prompt=None,
    prompt_column_name=None,
    completion_column_name=None,
):
    dataset = datasets.load_dataset(dataset_path)

    def to_chat_format(sample):
        if custom_system_prompt:
            system_msg = {"role": "system", "content": custom_system_prompt}
        user_msg = {"role": "user", "content": sample[prompt_column_name]}
        assistant_msg = {"role": "assistant", "content": sample[completion_column_name]}
        return {
            "prompt": [system_msg, user_msg] if custom_system_prompt else [user_msg],
            "completion": (
                [system_msg, user_msg, assistant_msg]
                if custom_system_prompt
                else [user_msg, assistant_msg]
            ),
        }
    if prompt_column_name is not None and completion_column_name is not None:
        dataset = dataset.map(to_chat_format).remove_columns(
            [prompt_column_name, completion_column_name]
        )
    elif prompt_column_name is None and completion_column_name is None:
        pass
    else:
        raise ValueError(
            "both user_prompt_column and assistant_output_column need to be set or None at the same time"
        )
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

    print(generated[0])
    list_dataset = [
        {"completion": x[0]["generated_text"], "prompt": dataset["train"][i]["prompt"]}
        for i, x in enumerate(generated)
    ]
    idx = int(len(list_dataset) * 0.8)  # 80/20 split
    cached_train = Dataset.from_list(list_dataset[:idx])
    cached_train.to_json(train_path)
    cached_test = Dataset.from_list(list_dataset[idx : len(list_dataset)])
    cached_test.to_json(test_path)


def train_student_model(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_length=512,
    use_lora=False,
):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_fc", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

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
        pad_token=tokenizer.pad_token,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=lora_config if use_lora else None,
    )

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
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequence = out_ids[0].tolist()
    if tokenizer.convert_tokens_to_ids("</s>") in sequence:
        cut_at = sequence.index(tokenizer.eos_token_id)
        sequence = sequence[: cut_at + 1]

    return tokenizer.decode(sequence)


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
    use_lora = args.use_lora
    learning_rate = float(args.learning_rate)
    per_device_train_batch_size = int(args.per_device_train_batch_size)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    num_train_epochs = int(args.num_train_epochs)
    max_length = int(args.max_length)

    # Meta
    train_path = args.train_dataset_path
    test_path = args.test_dataset_path
    verbose = args.verbose
    tuned_weights_target_path = args.tuned_weights_target_path
    
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
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.info("No cached targets found, generating teacher responses")
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(DEVICE)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)

        if teacher_tokenizer.pad_token is None:
            logging.info("Pad token not found for teacher tokenizer, setting to [PAD]")
            teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            teacher_model.resize_token_embeddings(len(teacher_tokenizer))

        generate_hard_targets(
            teacher_model,
            teacher_tokenizer,
            dataset_path=dataset_path,
            max_new_tokens=max_new_tokens,
            batch_size=inference_batch_size,
            custom_system_prompt=custom_system_prompt,
            prompt_column_name=user_column_name,
            completion_column_name=assistant_column_name,
            train_path=train_path,
            test_path=test_path,
        )
        del teacher_model, teacher_tokenizer
    else:
        logging.info(f"Responses already cached, using {train_path} and {test_path}")

    # Student model learning
    # Initialization
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(student_model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    trl.clone_chat_template(
        model, tokenizer, source_tokenizer_path=teacher_model_path
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    sample_teacher_response = tokenizer.apply_chat_template(test_dataset[0]["completion"], tokenize=False)
    sample_before_response = generate_from_prompt(
        test_dataset[0]["prompt"], 
        tokenizer, 
        model,
        max_new_tokens=max_new_tokens,
    )

    # Training
    logging.info("Training student model")
    model, tokenizer = train_student_model(
        model,
        tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        use_lora=use_lora,
    )

    # Save weights
    logging.info(f"Saving model to {tuned_weights_target_path}")
    model.save_pretrained(tuned_weights_target_path)
    tokenizer.save_pretrained(tuned_weights_target_path)
    
    # Show sample completions
    sample_after_response = generate_from_prompt(
        test_dataset[0]["prompt"],
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
    tuned_weights_path = args.tuned_weights_path

    if not os.path.exists(tuned_weights_path):
        logging.error(f"No model found in {tuned_weights_path}")
        return

    model = AutoModelForCausalLM.from_pretrained(tuned_weights_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(tuned_weights_path)

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
        description="Run Microbrewery (modes: distill, gen)"
    )
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Available modes"
    )

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
        "--use-lora", action="store_true", help="Enable LoRA (flag)"
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
        "--train-dataset-path",
        default=DEFAULT_TRAIN_PATH,
        help="Path to save/load cached train dataset JSON",
    )
    p_distill.add_argument(
        "--test-dataset-path",
        default=DEFAULT_TEST_PATH,
        help="Path to save/load cached test dataset JSON",
    )
    p_distill.add_argument(
        "--tuned-weights-target-path",
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
        "--tuned-weights-path", 
        required=True, 
        help="Path to a folder containing distilled model's weights"
    )
    p_infer.set_defaults(func=infer)

    args = parser.parse_args()

    # TODO: make it not break infer
    # sanity check: user-column only makes sense if assistant-column was provided
    # if (
        # args.func == distill
        # and (args.user_column_name and not args.assistant_column_name)
        # or (args.assistant_column_name and not args.user_column_name)
    # ):
        # p_distill.error(
            # "both --user-column-name and --assistant-column-name need to be set or None at the same time"
        # )

    args.func(args)


if __name__ == "__main__":
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
    main()
