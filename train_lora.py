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


# Comprehensive text cleaning function
def clean_text(text):
    # Replace common encoding artifacts
    replacements = {
        "â€™": "'",  # Apostrophe
        "â€”": "—",  # Em dash
        "â€¦": "…",  # Ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Fix all common contractions
    contractions = {
        "Whatâ€™s": "What's",
        "Howâ€™s": "How's",
        "donâ€™t": "don't",
        "itâ€™s": "it's",
        "youâ€™re": "you're",
        "Iâ€™ll": "I'll",
        "wonâ€™t": "won't",
        "wouldnâ€™t": "wouldn't",
        "youâ€™ve": "you've",
        "Iâ€™m": "I'm",
        "heâ€™s": "he's",
        "sheâ€™s": "she's",
        "theyâ€™re": "they're",
        "weâ€™re": "we're",
        "canâ€™t": "can't",
        "isnâ€™t": "isn't",
        "arenâ€™t": "aren't",
        "havenâ€™t": "haven't",
        "hasnâ€™t": "hasn't",
        "letâ€™s": "let's",
        "thatâ€™s": "that's",
        "thereâ€™s": "there's",
        "Iâ€™d": "I'd",
        "Iâ€™ve": "I've",
        "sheâ€™ll": "she'll",
        "heâ€™ll": "he'll",  # Adding just in case
        "weâ€™ll": "we'll",  # Adding just in case
        "theyâ€™ll": "they'll",  # Adding just in case
    }
    with open("dataset.txt", "r") as f:
        data = f.read()
    for old, new in contractions.items():
        if old in data:
            print(f"Found {old} -> {new}")

    # Optional: Convert "you're" to "your" before nouns (uncomment if needed)
    # import re
    # text = re.sub(r"you're\s+(\w+)", r"your \1", text)  # e.g., "you're day" -> "your day"

    return text


# Load tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, token=os.getenv("HUGGINGFACE_TOKEN")
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add special tokens
print("Tokenizing [INST]:", tokenizer.encode("[INST]"))
if len(tokenizer.encode("[INST]", add_special_tokens=False)) > 1:
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
if tokenizer.get_added_vocab():
    model.resize_token_embeddings(len(tokenizer))

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# Load and clean dataset
with open("dataset.txt", "r") as f:
    data = f.read().split("###")[1:]
examples = [entry.strip().split("\nOutput: ") for entry in data]
inputs = [clean_text(e[0].replace("Input: ", "")) for e in examples]
outputs = [clean_text(e[1]) for e in examples]

print("Inputs:", inputs)
print("Outputs:", outputs)

if len(inputs) != len(outputs):
    raise ValueError(
        f"Length mismatch: {len(inputs)} inputs and {len(outputs)} outputs."
    )

# Combine inputs and outputs
combined_texts = [
    f"[INST] {inp} [/INST] {out}</s>" for inp, out in zip(inputs, outputs)
]

# Tokenize
encodings = tokenizer(
    combined_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = input_ids.clone()

# Trim leading </s> tokens (safety net)
eos_token_id = tokenizer.eos_token_id
for i in range(len(input_ids)):
    tokens = input_ids[i].tolist()
    start_idx = 0
    while start_idx < len(tokens) and tokens[start_idx] == eos_token_id:
        start_idx += 1
    if start_idx > 0:
        input_ids[i, :start_idx] = tokenizer.pad_token_id
        attention_mask[i, :start_idx] = 0
        labels[i, :start_idx] = -100

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
