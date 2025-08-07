import argparse
import logging

from microbrewery.distiller import distill, finetune, infer


def set_verbose(flag: bool):
    logging.basicConfig(
        level=logging.DEBUG if flag else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def finetune_(args):
    student_model_path = args.student_model_path
    dataset_path = args.dataset_pathh
    custom_system_prompt = args.custom_system_prompt
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

    set_verbose(verbose)

    finetune(
        student_model_path=student_model_path,
        dataset_path=dataset_path,
        custom_system_prompt=custom_system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
        completion_column_name=completion_column_name,
        prompt_column_name=prompt_column_name,
        max_new_tokens=max_new_tokens,
        lora_targets=lora_targets,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        output_dir=output_dir
    )    


def distill_(args):
    # General settings
    teacher_model_path = args.teacher_model_path
    student_model_path = args.student_model_path
    dataset_path = args.dataset_path
    custom_system_prompt = args.custom_system_prompt

    # Teacher model
    max_new_tokens = int(args.max_new_tokens)
    teacher_batch_size = int(args.teacher_batch_size)
    completion_column_name = args.completion_column_name
    prompt_column_name = args.prompt_column_name
    num_sequences = int(args.num_sequences)

    # Student model
    lora_targets = args.lora_targets.split(",") if args.lora_targets is not None else None
    learning_rate = float(args.learning_rate)
    per_device_train_batch_size = int(args.per_device_train_batch_size)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    num_train_epochs = int(args.num_train_epochs)
    training_max_tokens = int(args.training_max_tokens)

    # Meta
    cached_targets_path = args.cached_targets_path
    verbose = args.verbose
    output_dir = args.output_dir

    set_verbose(verbose)

    distill(
        teacher_model_path=teacher_model_path,
        student_model_path=student_model_path,
        dataset_path=dataset_path,
        custom_system_prompt=custom_system_prompt,
        max_new_tokens=max_new_tokens,
        teacher_batch_size=teacher_batch_size,
        completion_column_name=completion_column_name,
        prompt_column_name=prompt_column_name,
        num_sequences=num_sequences,
        lora_targets=lora_targets,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        training_max_tokens=training_max_tokens,
        cached_targets_path=cached_targets_path,
        output_dir=output_dir
    )

def infer_(args):
    system_prompt = args.system_prompt
    user_prompt = args.user_prompt
    model_path = args.model_path

    print(
        infer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=model_path,
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
        "--student-model-path", required=True, help="Name of the student model"
    )
    p_ft.add_argument(
        "--dataset-path", required=True, help="Name of the dataset"
    )
    p_ft.add_argument(
        "--chat-template-tokenizer", default=None, help="If the current model is not conversational, clone this chat template"
    )
    p_ft.add_argument(
        "--custom-system-prompt", default=None, help="System prompt text"
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
    p_ft.set_defaults(func=finetune_)

    ## Distillation mode ##
    p_distill = subparsers.add_parser(
        "distill", help="Distill teacher model's knowledge into student model's weights"
    )
    p_distill.add_argument(
        "--teacher-model-path", required=True, help="Name of the teacher model"
    )
    p_distill.add_argument(
        "--student-model-path", required=True, help="Name of the student model"
    )
    p_distill.add_argument(
        "--dataset-path", required=True, help="Name of the dataset"
    )
    p_distill.add_argument(
        "--custom-system-prompt", required=True, help="System prompt text"
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
        "--training-max-tokens", default=256, help="Max length of prompt + completion in tokens (used when training)"
    )
    p_distill.add_argument(
        "--max-new-tokens",
        default=128,
        help="Max new tokens generated by model (used when generating, including targets by teacher model)",
    )
    p_distill.add_argument(
        "--teacher-batch-size",
        default=4,
        help="Batch size when generating teacher targets",
    )
    p_distill.add_argument(
        "--num-sequences",
        default=1,
        help="How many sequences to generate by teacher model",
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
        "--completion-column-name",
        default=None,
        help="Name of the completion column (optional, only for Q&A datasets)",
    )
    p_distill.add_argument(
        "--prompt-column-name",
        default=None,
        help="Name of the prompt column (optional; only used if --completion-column-name is set)",
    )
    p_distill.set_defaults(func=distill_)

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
    p_infer.set_defaults(func=infer_)

    args = parser.parse_args()

    args.func(args)

    
main()
