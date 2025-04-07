import subprocess
import os
from conversation import tokenizer, model  # Use CodeGen from conversation.py
from offline_tools import save_command as save_command_to_db

ALLOWED_LANGUAGES = ["cmd", "ps1", "python"]

def generate_code(language, task):
    if language not in ALLOWED_LANGUAGES:
        return f"Sorry, I only support {', '.join(ALLOWED_LANGUAGES)}."
    prompt = f"# {language} script to {task}\n"
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code_lines = code.split("\n")
        cleaned_code = "\n".join(
            line for line in code_lines if not line.strip().startswith("#") and len(line.strip()) > 0
        )
        return cleaned_code.strip()
    except Exception as e:
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
    prompt = f"Explain {concept} in {language} with an example"
    try:
        explanation = generator(prompt, max_length=150, num_return_sequences=1)[0][
            "generated_text"
        ]
        return explanation.strip()
    except Exception as e:
        return f"Explanation error: {str(e)}"

def save_command(name, language=None, code=None):
    if language and code:
        save_command_to_db(name, language, code)
        return f"Command '{name}' saved!"
    return "No code to save yet."
