from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import torch
import os

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, token=os.getenv("HUGGINGFACE_TOKEN")
)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos token

# Optionally add special tokens if not present
special_tokens = {"additional_special_tokens": ["[INST]", "[/INST]"]}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    ),
    device_map="auto",
    torch_dtype=torch.float16,
    token=os.getenv("HUGGINGFACE_TOKEN"),
)
model.resize_token_embeddings(len(tokenizer))  # Adjust for added tokens

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# Load dataset
with open("dataset.txt", "r") as f:
    data = f.read().split("###")[1:]  # Skip empty first split
examples = [entry.strip().split("\nOutput: ") for entry in data]
inputs = [e[0].replace("Input: ", "") for e in examples]
outputs = [e[1] for e in examples]

# Print for debugging
print("Inputs:", inputs)
print("Outputs:", outputs)

if len(inputs) != len(outputs):
    raise ValueError(
        f"Length mismatch: {len(inputs)} inputs and {len(outputs)} outputs."
    )

# Combine inputs and outputs
combined_texts = [
    f"<s>[INST] {inp} [/INST] {out}</s>" for inp, out in zip(inputs, outputs)
]

# Tokenize
encodings = tokenizer(
    combined_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = input_ids.clone()

# Mask input tokens in labels
inst_end_seq = tokenizer.encode("[/INST]", add_special_tokens=False)
for i, ids in enumerate(input_ids):
    tokens = ids.tolist()
    for pos in range(len(tokens) - len(inst_end_seq) + 1):
        if tokens[pos : pos + len(inst_end_seq)] == inst_end_seq:
            labels[i, : pos + len(inst_end_seq)] = -100
            break
    else:
        print(f"Warning: [/INST] not found in example {i}")
    if i == 0:
        print(f"Example input: {tokenizer.decode(tokens)}")
        print(f"Labels (first 20 tokens): {labels[i, :20].tolist()}")

# Debug shapes
print(f"Input IDs shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")


# Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


train_dataset = CustomDataset(input_ids, attention_mask, labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train and save
trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
