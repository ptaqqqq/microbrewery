import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
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


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device {_DEVICE}")


def pc_to_conversational_pc(
    dataset: str,
    prompt_column_name: str,
    completion_column_name: str,
    custom_system_prompt: str,
):
    def to_chat_format(sample):
        if custom_system_prompt:
            system_msg = {"role": "system", "content": custom_system_prompt}
        user_msg = {"role": "user", "content": sample[prompt_column_name]}
        assistant_msg = {"role": "assistant", "content": sample[completion_column_name]}
        return {
            "prompt": [system_msg, user_msg] if custom_system_prompt else [user_msg],
            "completion": [assistant_msg],
        }

    if prompt_column_name is not None and completion_column_name is not None:
        return dataset.map(
            to_chat_format, remove_columns=[prompt_column_name, completion_column_name]
        )
    else:
        raise ValueError(
            "prompt_column_name and completion_column_name are both required fields"
        )


def generate_hard_targets(
    teacher_model,
    teacher_tokenizer,
    dataset_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    num_return_sequences: int = 3,
    custom_system_prompt: Optional[str] = None,
    prompt_column_name: Optional[str] = None,
    completion_column_name: Optional[str] = None,
):
    dataset = datasets.load_dataset(dataset_path)

    # Use `or` instead of `and` so that the next function raises an exception if only one is set
    if prompt_column_name or completion_column_name:
        dataset = pc_to_conversational_pc(
            dataset, prompt_column_name, completion_column_name, custom_system_prompt
        )
    train_dataset = KeyDataset(dataset["train"], "prompt")

    assert all(
        m["content"] is not None for chat in train_dataset for m in chat
    ), "Found a None content in chats"

    logging.debug(f"pad {teacher_tokenizer.pad_token_id}")
    logging.debug(f"eos {teacher_tokenizer.eos_token_id}")

    pipe = TextGenerationPipeline(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        device=_DEVICE,
        pad_token_id=teacher_tokenizer.pad_token_id,
        eos_token_id=teacher_tokenizer.eos_token_id,
    )

    logging.info("Started pipeline")
    generated = pipe(
        train_dataset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )
    logging.info("Finished pipeline")

    list_dataset = []
    for i, out_list in enumerate(generated):
        prompt = dataset["train"][i]["prompt"]
        for sample in out_list:
            text = [sample["generated_text"][-1]]
            list_dataset.append({"prompt": prompt, "completion": text})

    idx = int(len(list_dataset) * 0.8)  # 80/20 split
    targets_train = Dataset.from_list(list_dataset[:idx])
    targets_test = Dataset.from_list(list_dataset[idx : len(list_dataset)])

    return targets_train, targets_test


def train_student_model(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    output_dir: str,
    learning_rate: int = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    max_length: int = 512,
    lora_targets: Optional[List[str]] = None,
):
    # TODO add more options for the config as arguments
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


def generate_from_prompt(
    prompt: List[Dict[str, str]], tokenizer, model, max_new_tokens: int = 128
):
    inputs = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(inputs, return_tensors="pt").to(_DEVICE)
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
        sequence = sequence[: cut_at + 1]

    return tokenizer.decode(sequence)


def finetune(
    student_model_path: str,
    dataset_path: str,
    custom_system_prompt: Optional[str] = None,
    chat_template_tokenizer: Optional[str] = None,
    completion_column_name: Optional[str] = None,
    prompt_column_name: Optional[str] = None,
    # Inference
    max_new_tokens: int = 128,
    # Training
    lora_targets: Optional[List[str]] = None,
    learning_rate: int = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    training_max_tokens: int = 256,
    # Meta
    output_dir: str = "./distilled-model",
):
    dataset = load_dataset(dataset_path)

    if completion_column_name or prompt_column_name:
        dataset = pc_to_conversational_pc(
            dataset, prompt_column_name, completion_column_name, custom_system_prompt
        )

    model = AutoModelForCausalLM.from_pretrained(student_model_path).to(_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)

    if chat_template_tokenizer:
        model, tokenizer, _ = trl.clone_chat_template(
            model, tokenizer, source_tokenizer_path=chat_template_tokenizer
        )

    if "test" not in dataset.column_names:
        # 75/25 split if not provided
        dataset = dataset["train"].train_test_split()

    targets_train = dataset["train"]
    targets_test = dataset["test"]

    sample_dataset_response = tokenizer.apply_chat_template(
        targets_test[0]["completion"], tokenize=False
    )
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
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=training_max_tokens,
        lora_targets=lora_targets,
    )
    logging.info("Finished training")

    # Show sample completions
    sample_after_response = generate_from_prompt(
        targets_test[0]["prompt"], tokenizer, model, max_new_tokens=max_new_tokens
    )
    print("Sample response generated by teacher:")
    print(sample_dataset_response)
    print("\nSample response generated by student before training:")
    print(sample_before_response)
    print("\nSample response generated by student after training:")
    print(sample_after_response)


def distill(
    teacher_model_path: str,
    student_model_path: str,
    dataset_path: str,
    custom_system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    # Teacher model
    teacher_batch_size: int = 1,
    completion_column_name: Optional[str] = None,
    prompt_column_name: Optional[str] = None,
    num_sequences: int = 1,
    # Student model
    lora_targets: Optional[List[str]] = None,
    learning_rate: int = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    training_max_tokens: int = 256,
    # Meta
    cached_targets_path: str = "./microbrewery-cached-targets",
    output_dir: str = "./distilled-model",
):
    logging.info("Starting distillation")

    # Hard target caching
    if cached_targets_path is None or not os.path.exists(cached_targets_path):
        logging.info("No cached targets found, generating teacher responses")
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(_DEVICE)
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
            batch_size=teacher_batch_size,
            num_return_sequences=num_sequences,
            custom_system_prompt=custom_system_prompt,
            prompt_column_name=prompt_column_name,
            completion_column_name=completion_column_name,
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
        dataset = load_dataset(
            "json", data_files={"train": str(train_path), "test": str(test_path)}
        )
        targets_train = dataset["train"]
        targets_test = dataset["test"]

    # Student model learning
    # Initialization
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(student_model_path).to(_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    model, tokenizer, _ = trl.clone_chat_template(
        model, tokenizer, source_tokenizer_path=teacher_model_path
    )

    sample_teacher_response = tokenizer.apply_chat_template(
        targets_test[0]["completion"], tokenize=False
    )
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
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=training_max_tokens,
        lora_targets=lora_targets,
    )
    logging.info("Finished training")

    # Show sample completions
    sample_after_response = generate_from_prompt(
        targets_test[0]["prompt"], tokenizer, model, max_new_tokens=max_new_tokens
    )
    print("Sample response generated by teacher:")
    print(sample_teacher_response)
    print("\nSample response generated by student before training:")
    print(sample_before_response)
    print("\nSample response generated by student after training:")
    print(sample_after_response)


def infer(
    system_prompt: str,
    user_prompt: str,
    model_path: str,
):
    if not os.path.exists(model_path):
        logging.error(f"No model found in {model_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path).to(_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return generate_from_prompt(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenizer=tokenizer,
        model=model,
    )
