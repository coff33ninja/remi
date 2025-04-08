# conversation.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from offline_tools import nlp, speak_response
from apis import API_KEYS
import re
import torch

# Load tokenizer and model from fine-tuned checkpoint
hf_token = API_KEYS.get("huggingface") or os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "Hugging Face token is missing. Please set it in the .env file."
    )

tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model", token=hf_token)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=hf_token,
)

model.resize_token_embeddings(len(tokenizer))
model.load_adapter("./fine_tuned_model")


def recognize_intent(command):
    doc = nlp(command.lower())
    entities = {ent.label_: ent.text for ent in doc.ents}
    command_lower = command.lower()

    if "switch to better model" in command_lower:
        return "switch_to_better_model", {}

    if " and " in command_lower or " then " in command_lower or " if " in command_lower:
        return parse_complex_command(command_lower, entities)

    if "code" in command_lower and (
        "write" in command_lower
        or "code a script" in command_lower
        or "create a code" in command_lower
    ):
        lang = next(
            (
                language
                for language in ["cmd", "ps1", "python"]
                if language in command_lower
            ),
            "python",
        )
        task = (
            command_lower.split(f"{lang} to")[-1].strip()
            if f"{lang} to" in command_lower
            else command_lower.replace("code a script for", "")
            .replace("write", "")
            .replace("create a code", "")
            .strip()
        )
        return "generate_code", {"language": lang, "task": task}

    if "feed specials from" in command_lower:
        filename = command_lower.split("from")[-1].strip()
        return "feed_specials_from_file", {"filename": filename}
    elif "scrape specials from" in command_lower:
        store = command_lower.split("from")[-1].strip()
        return "scrape_store_specials", {"store": store}
    elif "where can i buy" in command_lower:
        item = command_lower.split("buy")[-1].strip()
        return "get_specials", {"item": item}
    elif "nearest" in command_lower and "buy" in command_lower:
        item = command_lower.split("buy")[-1].strip()
        user_lat, user_lon = -33.9249, 18.4241  # Cape Town
        return "find_nearest_special", {
            "item": item,
            "user_lat": user_lat,
            "user_lon": user_lon,
        }
    elif "i have r" in command_lower and "need" in command_lower:
        budget = float(re.search(r"r(\d+\.?\d*)", command_lower).group(1))
        items = command_lower.split("need")[-1].strip().split(" and ")
        return "find_items_within_budget", {"budget": budget, "items": items}

    if (
        "calculate" in command_lower
        or "what's" in command_lower
        or "add" in command_lower
        or "subtract" in command_lower
        or "multiply" in command_lower
        or "divide" in command_lower
    ):
        nums = re.findall(r"\d+", command_lower)
        if len(nums) >= 2:
            num1, num2 = nums[0], nums[1]
            if "add" in command_lower or "+" in command_lower:
                operation = "add"
            elif "subtract" in command_lower or "-" in command_lower:
                operation = "subtract"
            elif (
                "multiply" in command_lower
                or "times" in command_lower
                or "*" in command_lower
            ):
                operation = "multiply"
            elif "divide" in command_lower or "/" in command_lower:
                operation = "divide"
            else:
                operation = "add"
            return "calculate", {"operation": operation, "num1": num1, "num2": num2}
        return "generate_response", {
            "prompt": "Please provide two numbers to calculate."
        }

    if "weather" in command_lower:
        city = entities.get("GPE", "London")
        return "get_weather", {"city": city}
    elif "news" in command_lower:
        topic = entities.get(
            "TOPIC",
            (
                command_lower.split("news about")[-1].strip()
                if "news about" in command_lower
                else "world"
            ),
        )
        return "get_news", {"topic": topic}
    elif "translate" in command_lower:
        text = command_lower.split("to")[0].replace("translate", "").strip()
        target_lang = command_lower.split("to")[-1].strip()
        return "translate_text", {"text": text, "target_lang": target_lang}
    elif "image" in command_lower or "photo" in command_lower:
        query = command_lower.split("of")[-1].strip()
        return "search_images", {"query": query}
    elif "task" in command_lower or "todo" in command_lower:
        parts = command_lower.split("due")
        task = parts[0].replace("add task", "").replace("add todo", "").strip()
        due = parts[1].strip() if len(parts) > 1 else None
        return "add_task", {"task": task, "due": due}
    elif "trello" in command_lower:
        title = command_lower.split("card")[-1].strip()
        return "create_trello_card", {"title": title, "desc": "Created by assistant"}
    elif "slack" in command_lower:
        text = command_lower.split("to slack")[-1].strip()
        return "send_slack_message", {"channel": "#general", "text": text}
    elif "discord" in command_lower:
        text = command_lower.split("to discord")[-1].strip()
        return "send_discord_message", {
            "channel_id": "your_channel_id_here",
            "text": text,
        }
    elif "whatsapp" in command_lower:
        parts = command_lower.split("to")
        if len(parts) > 1:
            phone_part = parts[1]
            if "say" in phone_part:
                phone = phone_part.split("say")[0].strip()
                message = phone_part.split("say")[-1].strip()
            elif "tell" in phone_part:
                phone = phone_part.split("tell")[0].strip()
                message = phone_part.split("tell")[-1].strip()
            else:
                phone = "+1234567890"
                message = parts[-1].strip()
        else:
            phone = "+1234567890"
            message = parts[-1].strip()
        return "send_whatsapp_message", {"phone": phone, "message": message}
    elif "what" in command_lower or "how" in command_lower:
        return "query_wolfram", {"query": command}
    elif "directions" in command_lower:
        parts = command_lower.split("from")[-1].split("to")
        origin, destination = parts[0].strip(), parts[1].strip()
        return "get_directions", {"origin": origin, "destination": destination}
    elif "run" in command_lower or "execute" in command_lower:
        lang = next(
            (
                language
                for language in ["cmd", "ps1", "python"]
                if language in command_lower
            ),
            None,
        )
        code = (
            command_lower.split("code")[-1].strip() if "code" in command_lower else None
        )
        return "execute_code", {"language": lang, "code": code}
    elif "explain" in command_lower:
        lang = next(
            (
                language
                for language in ["cmd", "ps1", "python"]
                if language in command_lower
            ),
            "python",
        )
        concept = command_lower.split(f"in {lang}")[-1].strip()
        return "explain_concept", {"language": lang, "concept": concept}
    elif "save" in command_lower and "as" in command_lower:
        name = command_lower.split("as")[-1].strip()
        return "save_command", {"name": name}
    elif "contact" in command_lower and "add" in command_lower:
        name = entities.get(
            "PERSON", command_lower.split("contact")[-1].split("with")[0].strip()
        )
        phone = next((e.text for e in doc if e.text.startswith("+")), None)
        email = next((e.text for e in doc if "@" in e.text), None)
        address = re.search(r"address ([\w\s,]+)", command_lower)
        birthday = entities.get("DATE", None)
        notes = re.search(r"notes ([\w\s]+)", command_lower)
        category = re.search(r"category (\w+)", command_lower)
        return "add_contact", {
            "name": name,
            "phone": phone,
            "email": email,
            "address": address.group(1) if address else None,
            "birthday": birthday,
            "notes": notes.group(1) if notes else None,
            "category": category.group(1) if category else None,
        }
    elif "contact" in command_lower and (
        "show" in command_lower or "get" in command_lower
    ):
        name = entities.get("PERSON", command_lower.split("contact")[-1].strip())
        return "get_contact", {"name": name}
    elif "create database" in command_lower:
        db_name = command_lower.split("database")[-1].strip()
        return "create_database", {"db_name": db_name}
    elif "add to" in command_lower:
        parts = command_lower.split("add to")
        db_name = parts[1].split(" ")[0].strip()
        rest = parts[1].split(" ", 1)[1].strip()
        key = rest.split(" ")[0]
        value_match = re.search(r"value ([\w\s]+)", rest)
        tags_match = re.search(r"tags ([\w\s,]+)", rest)
        priority_match = re.search(r"priority (\d+)", rest)
        return "add_to_database", {
            "db_name": db_name,
            "key": key,
            "value": value_match.group(1) if value_match else rest.split(" ", 1)[1],
            "tags": tags_match.group(1) if tags_match else None,
            "priority": int(priority_match.group(1)) if priority_match else None,
        }
    elif "get from" in command_lower:
        parts = command_lower.split("get from")
        db_name = parts[1].split(" ")[0].strip()
        key = parts[1].split(" ", 1)[1].strip()
        return "get_from_database", {"db_name": db_name, "key": key}
    elif "volume" in command_lower:
        if "mute" in command_lower:
            return "set_volume", {"level": "mute"}
        elif "unmute" in command_lower:
            return "set_volume", {"level": "unmute"}
        else:
            level = re.search(r"volume to (\d+)", command_lower)
            return "set_volume", {"level": level.group(1) if level else "50"}
    elif "open" in command_lower and "app" in command_lower:
        app_name = command_lower.split("open")[-1].split("app")[-1].strip()
        return "open_app", {"app_name": app_name}
    elif "type" in command_lower:
        text = command_lower.split("type")[-1].strip()
        return "type_text", {"text": text}
    elif "game info" in command_lower:
        game = command_lower.split("info about")[-1].strip()
        return "get_game_info", {"game": game}
    elif "game news" in command_lower:
        game = command_lower.split("news about")[-1].strip()
        return "get_game_news", {"game": game}
    elif "game code" in command_lower:
        task = command_lower.split("code for")[-1].strip()
        return "generate_code", {
            "language": "python",
            "task": f"create a simple {task} game",
        }
    elif "create file" in command_lower:
        filename = command_lower.split("file")[-1].strip()
        return "create_file", {"filename": filename}
    elif "delete" in command_lower and (
        "file" in command_lower or "folder" in command_lower
    ):
        filename = (
            command_lower.split("delete")[-1]
            .split("file")[-1]
            .split("folder")[-1]
            .strip()
        )
        return "delete_file", {"filename": filename}
    elif "move" in command_lower and "to" in command_lower:
        parts = command_lower.split("to")
        src = parts[0].replace("move", "").replace("file", "").strip()
        dest = parts[1].strip()
        return "move_file", {"src": src, "dest": dest}
    elif "cpu usage" in command_lower:
        return "get_cpu_usage", {}
    elif "memory" in command_lower:
        return "get_memory_usage", {}
    elif "disk space" in command_lower:
        drive = re.search(r"on (\w:)", command_lower)
        return "get_disk_space", {"drive": drive.group(1) if drive else "C:"}
    elif "click at" in command_lower:
        coords = re.search(r"at (\d+) (\d+)", command_lower)
        x, y = coords.groups() if coords else (500, 500)
        return "click_at", {"x": x, "y": y}
    elif "find" in command_lower and "click" in command_lower:
        button_name = command_lower.split("button")[-1].split("and")[0].strip()
        return "find_and_click", {"button_name": button_name}
    elif "feed data from" in command_lower:
        filename = command_lower.split("from")[-1].strip()
        return "feed_data_from_file", {"filename": filename}
    elif "research" in command_lower and "save" in command_lower:
        topic = command_lower.split("research")[-1].split("and")[0].strip()
        return "research_and_save", {"topic": topic}
    elif "get research" in command_lower:
        topic = command_lower.split("about")[-1].strip()
        return "get_research", {"topic": topic}
    elif "tell me about myself" in command_lower or "analyze me" in command_lower:
        return "analyze_personality", {}
    else:
        return "generate_response", {"prompt": command}


def parse_complex_command(command, entities):
    if " if " in command:
        condition, action = command.split(" if ", 1)
        return "complex_if", {"condition": condition.strip(), "action": action.strip()}
    parts = re.split(r" and | then ", command)
    return "complex_chain", {"commands": [part.strip() for part in parts]}


def generate_response(prompt):
    try:
        full_prompt = f"A flirty and helpful assistant says: {prompt}"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(full_prompt):
            response = response[len(full_prompt) :].strip()
        return response
    except Exception as e:
        return f"Response generation error: {str(e)}"


def switch_to_better_model():
    speak_response(
        "You're already on Mistral 7B—pretty spicy already! Need a bigger flirt?"
    )
    return "Currently on Mistral 7B—no better model available yet for your hardware."


def switch_to_better_conversational_model():
    speak_response("Switching to a better conversational model—hold tight!")
    return "Switching conversational models not implemented yet."
