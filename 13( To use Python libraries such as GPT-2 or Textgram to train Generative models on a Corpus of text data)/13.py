
#before running, in terminal: pip install tf-keras and pip install accelerate
import os
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb

# ----------------------------------------
# Step 1: Load Tokenizer and Model
# ----------------------------------------
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# GPT-2 has no pad token → set EOS as pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ----------------------------------------
# Step 2: Load Dataset (local file)
# ----------------------------------------
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "mycorpus.txt"})

# ----------------------------------------
# Step 3: Tokenize Dataset
# ----------------------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

tokenized_dataset = tokenized_dataset.map(
    lambda x: {"labels": x["input_ids"]},
    batched=True
)

tokenized_dataset.set_format("torch")

# ----------------------------------------
# Step 4: Training
# ----------------------------------------
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

# ----------------------------------------
# Step 5: Generate Text
# ----------------------------------------
prompt = "Artificial intelligence is"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=120,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
