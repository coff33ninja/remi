from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import os

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16  # Match compute dtype with model dtype
    ),
    device_map="auto",
    torch_dtype=torch.float16,
    token=os.getenv("HUGGINGFACE_TOKEN"),
)

# Configure LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

# Load dataset
with open("dataset.txt", "r") as f:
    data = f.read().split("###")[1:]  # Skip empty first split
examples = [entry.strip().split("\nOutput: ") for entry in data]
inputs = [e[0].replace("Input: ", "") for e in examples]
outputs = [e[1] for e in examples]

# Print inputs and outputs for debugging
print("Inputs:", inputs)
print("Outputs:", outputs)

# Check lengths of inputs and outputs
if len(inputs) != len(outputs):
    raise ValueError(f"Length mismatch: {len(inputs)} inputs and {len(outputs)} outputs.")

# Combine inputs and outputs using Mistral's instruction format
combined_texts = []
for inp, out in zip(inputs, outputs):
    # Format following Mistral's instruction format
    prompt = f"<s>[INST] {inp} [/INST] {out}</s>"
    combined_texts.append(prompt)

# Tokenize the combined texts
encodings = tokenizer(combined_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]

# For causal language modeling, labels are the same as input_ids
# But we set -100 for tokens we don't want to predict (the input part)
labels = input_ids.clone()

# Find positions of [/INST] tokens to separate input from output
for i, ids in enumerate(input_ids):
    # Convert to list for easier debugging
    tokens = ids.tolist()
    # Find the position of [/INST] token or its ID
    inst_token_id = tokenizer.encode(" [/INST]", add_special_tokens=False)[-1]  # Get the last token ID
    inst_positions = [pos for pos, token_id in enumerate(tokens) if token_id == inst_token_id]
    
    if inst_positions:
        # Set all tokens before [/INST] to -100 (don't predict these)
        inst_pos = inst_positions[-1]  # Take the last occurrence if multiple
        labels[i, :inst_pos+1] = -100  # +1 to include the [/INST] token itself
    
    # Print for debugging (first example only)
    if i == 0:
        print(f"Example input: {tokenizer.decode(tokens)}")
        print(f"[/INST] token found at position: {inst_positions if inst_positions else 'Not found'}")

# Debug: Print shapes to verify alignment
print(f"Input IDs shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")

# Prepare dataset
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
            "labels": self.labels[idx]
        }

train_dataset = CustomDataset(input_ids, attention_mask, labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced batch size for GTX 1060 6GB
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
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
