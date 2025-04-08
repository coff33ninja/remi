import subprocess
import os
import logging
import tempfile  # Add this import
from datetime import datetime  # Add this import

import torch
from conversation import tokenizer, model
from offline_tools import save_command as save_command_to_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

ALLOWED_LANGUAGES = ["cmd", "ps1", "python"]

if model is None or tokenizer is None:
    raise RuntimeError("Model or tokenizer not loaded from conversation.py")


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
    prompt = f"A helpful assistant writes a {language} script to {task} with detailed comments and error handling:\n```{language}\n"
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
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Raw generated code:\n{code}")

        code_block = (
            code.split(f"```{language}")[-1].split("```")[0].strip()
            if f"```{language}" in code
            else code
        )
        final_code = "\n".join(
            line for line in code_block.split("\n") if len(line.strip()) > 0
        )
        logging.info(f"Cleaned code:\n{final_code}")

        # Prompt user to save the code
        save_prompt = input("Do you want to save this code? (yes/no): ").strip().lower()
        if save_prompt == "yes":
            description = input("Enter a description for the code: ").strip()
            save_generated_code(language, final_code, description)

        return final_code
    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return f"Code generation error: {str(e)}"


def execute_code(language, code):
    """
    Execute the provided code in the specified language.

    Args:
        language (str): Programming language of the code
        code (str): Code to execute

    Returns:
        str: Execution output or error message
    """
    if language not in ALLOWED_LANGUAGES:
        return f"Unsupported language: {language}"
    extension = {"cmd": ".bat", "ps1": ".ps1", "python": ".py"}[language]
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        temp_file.write(code.encode())
        temp_file.close()
        try:
            if language == "cmd":
                result = subprocess.run(
                    ["cmd.exe", "/c", temp_file.name], capture_output=True, text=True, timeout=10
                )
            elif language == "ps1":
                result = subprocess.run(
                    ["powershell.exe", "-File", temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            elif language == "python":
                result = subprocess.run(
                    ["python", temp_file.name], capture_output=True, text=True, timeout=10
                )
            return f"Output:\n{result.stdout}\nErrors (if any):\n{result.stderr}"
        finally:
            os.remove(temp_file.name)


def explain_concept(language, concept):
    """
    Explain a programming concept in the specified language with an example.

    Args:
        language (str): Programming language to use
        concept (str): Concept to explain

    Returns:
        str: Explanation or error message
    """
    if language not in ALLOWED_LANGUAGES:
        return f"Sorry, I only explain {', '.join(ALLOWED_LANGUAGES)}."
    prompt = f"A flirty assistant explains {concept} in {language} with a simple example:\n```{language}\n"
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

        explanation_block = (
            explanation.split(f"```{language}")[-1].split("```")[0].strip()
            if f"```{language}" in explanation
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
    """
    Save a generated code snippet to the database.

    Args:
        name (str): Name to save the command under
        language (str, optional): Programming language of the code
        code (str, optional): Code to save

    Returns:
        str: Success or error message
    """
    if language and code:
        save_command_to_db(name, language, code)
        return f"Command '{name}' saved!"
    return "No code to save yet."


def save_generated_code(language, code, description):
    """
    Save generated code to the database with metadata.

    Args:
        language (str): Programming language of the code.
        code (str): The generated code snippet.
        description (str): Description of the code.

    Returns:
        str: Success or error message.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_command_to_db(description, language, code)
    return f"Code saved successfully with description: '{description}' at {timestamp}."
