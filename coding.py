import subprocess
import os
import logging

import torch
from conversation import tokenizer, model  # Use Mistral 7B from conversation.py
from offline_tools import save_command as save_command_to_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

ALLOWED_LANGUAGES = ["cmd", "ps1", "python"]


def generate_code(language, task):
    """
    Generate code in the specified language based on the task description.
    
    Args:
        language (str): Programming language to generate code in (must be in ALLOWED_LANGUAGES)
        task (str): Description of what the code should do
        
    Returns:
        str: Generated code or error message
    """
    if language not in ALLOWED_LANGUAGES:
        return f"Sorry, I only support {', '.join(ALLOWED_LANGUAGES)}."
    prompt = f"A helpful assistant writes a {language} script to {task} with detailed comments and error handling:\n```python\n"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        outputs = model.generate(
            **inputs,
            max_length=400,  # Increased for more complete scripts
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Raw generated code:\n{code}")

        # Extract code block and clean
        code_block = (
            code.split("```python")[-1].split("```")[0].strip()
            if "```python" in code
            else code
        )
        final_code = "\n".join(
            line for line in code_block.split("\n") if len(line.strip()) > 0
        )
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
    prompt = f"A flirty assistant explains {concept} in {language} with a simple example:\n```python\n"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        outputs = model.generate(
            **inputs,
            max_length=400,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Raw explanation:\n{explanation}")

        # Extract and clean
        explanation_block = (
            explanation.split("```python")[-1].split("```")[0].strip()
            if "```python" in explanation
            else explanation
        )
        final_explanation = "\n".join(
            line for line in explanation_block.split("\n") if len(line.strip()) > 0
        )
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
