import subprocess
import os
import logging
from conversation import tokenizer, model  # Use CodeGen from conversation.py
from offline_tools import save_command as save_command_to_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

ALLOWED_LANGUAGES = ["cmd", "ps1", "python"]


def generate_code(language, task):
    if language not in ALLOWED_LANGUAGES:
        return f"Sorry, I only support {', '.join(ALLOWED_LANGUAGES)}."
    prompt = f"# {language} script to {task}\n# Include error handling and comments\n"
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)  # Explicit truncation
        outputs = model.generate(
            **inputs,
            max_length=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Raw generated code:\n{code}")

        # Preserve comments that follow the initial prompt lines
        code_lines = code.split("\n")
        cleaned_code = []
        skip_initial_prompt = True
        for line in code_lines:
            stripped_line = line.strip()
            if skip_initial_prompt and stripped_line.startswith("#"):
                if stripped_line in prompt.split("\n"):
                    continue
                skip_initial_prompt = False
            if len(stripped_line) > 0:  # Keep non-empty lines, including comments
                cleaned_code.append(line)
        final_code = "\n".join(cleaned_code).strip()
        logging.info(f"Cleaned code:\n{final_code}")
        return final_code
    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return f"Code generation error: {str(e)}"


def execute_code(language, code):
    if language not in ALLOWED_LANGUAGES:
        return f"Unsupported language: {language}"
    extension = {"cmd": ".bat", "ps1": ".ps1", "python": ".py"}[language]
    filename = f"temp_script{extension}"
    try:
        with open(filename, "w") as f:
            f.write(code)
        if language == "cmd":
            result = subprocess.run(
                ["cmd.exe", "/c", filename], capture_output=True, text=True, timeout=10
            )
        elif language == "ps1":
            result = subprocess.run(
                ["powershell.exe", "-File", filename],
                capture_output=True,
                text=True,
                timeout=10,
            )
        elif language == "python":
            result = subprocess.run(
                ["python", filename], capture_output=True, text=True, timeout=10
            )
        os.remove(filename)
        return f"Output:\n{result.stdout}\nErrors (if any):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        os.remove(filename)
        return "Execution timed out."
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        return f"Execution failed: {str(e)}"


def explain_concept(language, concept):
    if language not in ALLOWED_LANGUAGES:
        return f"Sorry, I only explain {', '.join(ALLOWED_LANGUAGES)}."
    prompt = f"# Explain {concept} in {language} with an example\n"
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)  # Explicit truncation
        outputs = model.generate(
            **inputs,
            max_length=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Raw explanation:\n{explanation}")

        # Similar cleaning logic as generate_code
        explanation_lines = explanation.split("\n")
        cleaned_explanation = []
        skip_initial_prompt = True
        for line in explanation_lines:
            stripped_line = line.strip()
            if skip_initial_prompt and stripped_line.startswith("#"):
                if stripped_line in prompt.split("\n"):
                    continue
                skip_initial_prompt = False
            if len(stripped_line) > 0:
                cleaned_explanation.append(line)
        final_explanation = "\n".join(cleaned_explanation).strip()
        logging.info(f"Cleaned explanation:\n{final_explanation}")
        return final_explanation
    except Exception as e:
        logging.error(f"Explanation error: {str(e)}")
        return f"Explanation error: {str(e)}"


def save_command(name, language=None, code=None):
    if language and code:
        save_command_to_db(name, language, code)
        return f"Command '{name}' saved!"
    return "No code to save yet."
