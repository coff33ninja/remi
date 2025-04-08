import spacy
import sqlite3
from pocketsphinx import LiveSpeech
import pyttsx3
from datetime import datetime
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import subprocess
import pyautogui
import os
import shutil
import psutil
import pytesseract
from PIL import Image
import PyPDF2
import re
from calculations import haversine_distance
import snowboydecoder
import logging
import time
import hashlib
from cryptography.fernet import Fernet
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import win10toast
import schedule
import threading
import socket
import pyshark
import pyperclip
from PIL import ImageGrab

nlp = spacy.load("en_core_web_sm")

databases = {}


def get_db(db_name):
    if (db_name not in databases):
        conn = sqlite3.connect(f"{db_name}.db")
        databases[db_name] = {"conn": conn, "cursor": conn.cursor()}
    return databases[db_name]


core_db = get_db("core")
core_db["cursor"].execute(
    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
)
core_db["cursor"].execute(
    "CREATE TABLE IF NOT EXISTS commands (name TEXT PRIMARY KEY, language TEXT, code TEXT)"
)
core_db["cursor"].execute(
    """
    CREATE TABLE IF NOT EXISTS images (
        path TEXT PRIMARY KEY,
        description TEXT,
        timestamp TEXT
    )
    """
)
core_db["conn"].commit()

address_db = get_db("addressbook")
address_db["cursor"].execute(
    """
    CREATE TABLE IF NOT EXISTS contacts (
        name TEXT PRIMARY KEY,
        phone TEXT,
        email TEXT,
        address TEXT,
        birthday TEXT,
        notes TEXT,
        category TEXT
    )
"""
)

personality_db = get_db("personality")
personality_db["cursor"].execute(
    """
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        command TEXT,
        topic TEXT,
        timestamp TEXT
    )
"""
)
personality_db["cursor"].execute(
    """
    CREATE TABLE IF NOT EXISTS research (
        topic TEXT PRIMARY KEY,
        data TEXT,
        source TEXT,
        timestamp TEXT
    )
"""
)

specials_db = get_db("specials")
specials_db["cursor"].execute(
    """
    CREATE TABLE IF NOT EXISTS specials (
        item TEXT,
        price REAL,
        store TEXT,
        latitude REAL,
        longitude REAL,
        timestamp TEXT
    )
"""
)
specials_db["cursor"].execute("CREATE INDEX IF NOT EXISTS idx_item ON specials (item)")


def create_database(db_name):
    db = get_db(db_name)
    db["cursor"].execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            key TEXT PRIMARY KEY,
            value TEXT,
            timestamp TEXT,
            tags TEXT,
            priority INTEGER
        )
    """
    )
    db["conn"].commit()
    return f"Database '{db_name}' created!"


def cache_result(key, value, db_name="core"):
    db = get_db(db_name)
    db["cursor"].execute(
        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
    )
    db["conn"].commit()


def get_cached_result(key, db_name="core"):
    db = get_db(db_name)
    db["cursor"].execute("SELECT value FROM cache WHERE key = ?", (key,))
    result = db["cursor"].fetchone()
    return result[0] if result else None


def save_command(name, language, code, db_name="core"):
    db = get_db(db_name)
    db["cursor"].execute(
        "INSERT OR REPLACE INTO commands (name, language, code) VALUES (?, ?, ?)",
        (name, language, code),
    )
    db["conn"].commit()


def get_command(name, db_name="core"):
    db = get_db(db_name)
    db["cursor"].execute("SELECT language, code FROM commands WHERE name = ?", (name,))
    result = db["cursor"].fetchone()
    return result if result else None


def add_contact(
    name,
    phone=None,
    email=None,
    address=None,
    birthday=None,
    notes=None,
    category=None,
    db_name="addressbook",
):
    db = get_db(db_name)
    db["cursor"].execute(
        """
        INSERT OR REPLACE INTO contacts (name, phone, email, address, birthday, notes, category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (name, phone, email, address, birthday, notes, category),
    )
    db["conn"].commit()
    return f"Contact '{name}' added!"


def get_contact(name, db_name="addressbook"):
    db = get_db(db_name)
    db["cursor"].execute(
        "SELECT phone, email, address, birthday, notes, category FROM contacts WHERE name = ?",
        (name,),
    )
    result = db["cursor"].fetchone()
    if result:
        phone, email, address, birthday, notes, category = result
        return (
            f"Contact: {name}, Phone: {phone or 'N/A'}, Email: {email or 'N/A'}, "
            f"Address: {address or 'N/A'}, Birthday: {birthday or 'N/A'}, Notes: {notes or 'N/A'}, "
            f"Category: {category or 'N/A'}"
        )
    return f"No contact found for '{name}'."


def add_to_database(db_name, key, value, tags=None, priority=None):
    db = get_db(db_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db["cursor"].execute(
        """
        INSERT OR REPLACE INTO items (key, value, timestamp, tags, priority)
        VALUES (?, ?, ?, ?, ?)
    """,
        (key, value, timestamp, tags, priority),
    )
    db["conn"].commit()
    return f"Added '{key}' to '{db_name}'!"


def get_from_database(db_name, key):
    db = get_db(db_name)
    db["cursor"].execute(
        "SELECT value, timestamp, tags, priority FROM items WHERE key = ?", (key,)
    )
    result = db["cursor"].fetchone()
    if result:
        value, timestamp, tags, priority = result
        return (
            f"Value: {value}, Added: {timestamp}, Tags: {tags or 'None'}, "
            f"Priority: {priority or 'None'}"
        )
    return f"No entry found for '{key}' in '{db_name}'."


def create_file(filename):
    try:
        with open(filename, "w") as f:
            f.write("")
        return f"File '{filename}' created."
    except Exception as e:
        return f"File creation error: {str(e)}"


def delete_file(filename):
    try:
        if os.path.isdir(filename):
            shutil.rmtree(filename)
            return f"Folder '{filename}' deleted."
        else:
            os.remove(filename)
            return f"File '{filename}' deleted."
    except Exception as e:
        return f"Deletion error: {str(e)}"


def move_file(src, dest):
    try:
        shutil.move(src, dest)
        return f"Moved '{src}' to '{dest}'."
    except Exception as e:
        return f"Move error: {str(e)}"


def get_cpu_usage():
    try:
        return f"CPU Usage: {psutil.cpu_percent(interval=1)}%"
    except Exception as e:
        return f"CPU usage error: {str(e)}"


def get_memory_usage():
    try:
        mem = psutil.virtual_memory()
        return f"Memory Usage: {mem.percent}% ({mem.used / 1024**3:.2f} GB used of {mem.total / 1024**3:.2f} GB)"
    except Exception as e:
        return f"Memory usage error: {str(e)}"


def get_disk_space(drive):
    try:
        disk = psutil.disk_usage(drive)
        return f"Disk Space on {drive}: {disk.percent}% used ({disk.free / 1024**3:.2f} GB free of {disk.total / 1024**3:.2f} GB)"
    except Exception as e:
        return f"Disk space error: {str(e)}"


def click_at(x, y):
    try:
        pyautogui.click(int(x), int(y))
        return f"Clicked at ({x}, {y})."
    except Exception as e:
        return f"Click error: {str(e)}"


def find_and_click(button_name):
    try:
        location = pyautogui.locateCenterOnScreen(f"{button_name}.png")
        if location:
            pyautogui.click(location)
            return f"Clicked '{button_name}' button."
        return f"Button '{button_name}' not found."
    except Exception as e:
        return f"Button click error: {str(e)}"


def set_volume(level):
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        if level == "mute":
            volume.SetMute(1, None)
            return "Volume muted."
        elif level == "unmute":
            volume.SetMute(0, None)
            return "Volume unmuted."
        else:
            vol = max(0.0, min(1.0, float(level) / 100))
            volume.SetMasterVolumeLevelScalar(vol, None)
            return f"Volume set to {level}%."
    except Exception as e:
        return f"Volume control error: {str(e)}"


def open_app(app_name):
    try:
        subprocess.Popen(app_name)
        return f"Opened {app_name}."
    except Exception as e:
        return f"App opening error: {str(e)}"


def type_text(text):
    try:
        pyautogui.typewrite(text)
        pyautogui.press("enter")
        return f"Typed '{text}'."
    except Exception as e:
        return f"Typing error: {str(e)}"


def log_query(command, topic):
    db = get_db("personality")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db["cursor"].execute(
        "INSERT INTO queries (command, topic, timestamp) VALUES (?, ?, ?)",
        (command, topic, timestamp),
    )
    db["conn"].commit()


def save_research(topic, data, source):
    db = get_db("personality")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db["cursor"].execute(
        "INSERT OR REPLACE INTO research (topic, data, source, timestamp) VALUES (?, ?, ?, ?)",
        (topic, data, source, timestamp),
    )
    db["conn"].commit()
    return f"Research on '{topic}' saved from {source}."


def get_research(topic):
    db = get_db("personality")
    db["cursor"].execute(
        "SELECT data, source, timestamp FROM research WHERE topic = ?", (topic,)
    )
    result = db["cursor"].fetchone()
    if result:
        data, source, timestamp = result
        return f"Research on '{topic}': {data} (Source: {source}, Saved: {timestamp})"
    return f"No research found for '{topic}'."


def analyze_personality():
    db = get_db("personality")
    db["cursor"].execute("SELECT command, topic FROM queries")
    queries = db["cursor"].fetchall()
    if not queries:
        return "I don’t have enough data about you yet!"

    topics = [q[1] for q in queries if q[1]]
    topic_counts = {}
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    interests = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    analysis = "Based on your queries, you seem interested in:\n"
    for topic, count in interests:
        analysis += f"- {topic} ({count} queries)\n"
    if "code" in [q[0] for q in queries]:
        analysis += "You might enjoy problem-solving or programming.\n"
    if "game" in [q[0] for q in queries]:
        analysis += "You seem to enjoy gaming or game-related topics.\n"
    return analysis


def feed_data_from_file(filename):
    try:
        if filename.endswith(".txt"):
            with open(filename, "r") as f:
                data = f.read()
            topic = os.path.splitext(filename)[0]
            return save_research(topic, data, f"file: {filename}")
        elif filename.endswith(".pdf"):
            with open(filename, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                data = " ".join(page.extract_text() for page in pdf.pages)
            topic = os.path.splitext(filename)[0]
            return save_research(topic, data, f"file: {filename}")
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(filename)
            data = pytesseract.image_to_string(img)
            topic = os.path.splitext(filename)[0]
            return save_research(topic, data, f"file: {filename}")
        else:
            return "Unsupported file type."
    except Exception as e:
        return f"Data feed error: {str(e)}"


def feed_specials_from_file(filename):
    try:
        db = get_db("specials")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if filename.endswith(".txt"):
            with open(filename, "r") as f:
                lines = f.readlines()
            for line in lines:
                # Format: "item:price:store:lat:lon" (e.g., "milk:15.99:Shoprite:-33.9249:18.4241")
                parts = line.strip().split(":")
                if len(parts) == 5:
                    item, price, store, lat, lon = parts
                    db["cursor"].execute(
                        "INSERT INTO specials (item, price, store, latitude, longitude, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            item.lower(),
                            float(price),
                            store,
                            float(lat),
                            float(lon),
                            timestamp,
                        ),
                    )
            db["conn"].commit()
            return f"Specials from '{filename}' added to database."
        elif filename.endswith(".pdf"):
            with open(filename, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                text = " ".join(page.extract_text() for page in pdf.pages)
            # Use OCR if text is empty (scanned PDF)
            if not text.strip():
                images = [Image.open(f"page_{i}.png") for i in range(len(pdf.pages))]
                text = " ".join(pytesseract.image_to_string(img) for img in images)
                for i in range(len(pdf.pages)):
                    os.remove(f"page_{i}.png")
            # NLP-enhanced parsing
            doc = nlp(text)
            specials = []
            store = None
            lat, lon = None, None
            for sent in doc.sents:
                if "at" in sent.text.lower():
                    parts = sent.text.lower().split("at")
                    if len(parts) > 1:
                        item_price = parts[0].strip()
                        store_loc = parts[1].strip()
                        price_match = re.search(r"r(\d+\.?\d*)", item_price)
                        if price_match:
                            price = float(price_match.group(1))
                            item = re.sub(r"r\d+\.?\d*", "", item_price).strip()
                            store_match = re.search(
                                r"(shoprite|picknpay|checkers)", store_loc
                            )
                            store = (
                                store_match.group(0).capitalize()
                                if store_match
                                else "Unknown"
                            )
                            loc_match = re.search(
                                r"lat\s*(-?\d+\.?\d*)\s*lon\s*(-?\d+\.?\d*)", store_loc
                            )
                            lat = float(loc_match.group(1)) if loc_match else None
                            lon = float(loc_match.group(2)) if loc_match else None
                            specials.append((item, price, store, lat, lon))
            for item, price, store, lat, lon in specials:
                db["cursor"].execute(
                    "INSERT INTO specials (item, price, store, latitude, longitude, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (item, price, store, lat or 0, lon or 0, timestamp),
                )
            db["conn"].commit()
            return f"Specials from '{filename}' added to database."
        else:
            return "Unsupported file type for specials."
    except Exception as e:
        return f"Specials feed error: {str(e)}"


def get_specials(item):
    db = get_db("specials")
    db["cursor"].execute(
        "SELECT store, price, latitude, longitude FROM specials WHERE item = ?",
        (item.lower(),),
    )
    results = db["cursor"].fetchall()
    if results:
        return "\n".join(
            [
                f"{store}: R{price} (Lat: {lat}, Lon: {lon})"
                for store, price, lat, lon in results
            ]
        )
    return f"No specials found for '{item}'."


def find_nearest_special(item, user_lat, user_lon):
    db = get_db("specials")
    db["cursor"].execute(
        "SELECT store, price, latitude, longitude FROM specials WHERE item = ?",
        (item.lower(),),
    )
    results = db["cursor"].fetchall()
    if not results:
        return f"No specials found for '{item}'."

    nearest = None
    min_distance = float("inf")
    for store, price, lat, lon in results:
        if lat and lon:
            distance = haversine_distance(user_lat, user_lon, lat, lon)
            if isinstance(distance, float) and distance < min_distance:
                min_distance = distance
                nearest = (store, price, distance)

    if nearest:
        store, price, distance = nearest
        return f"Nearest: {store} at R{price}, {distance:.2f} km away."
    return "No location data available for specials."


def find_items_within_budget(budget, items):
    db = get_db("specials")
    item_prices = {}
    for item in items:
        db["cursor"].execute(
            "SELECT store, price, latitude, longitude FROM specials WHERE item = ?",
            (item.lower(),),
        )
        results = db["cursor"].fetchall()
        if results:
            item_prices[item] = {
                store: (price, lat, lon) for store, price, lat, lon in results
            }

    if len(item_prices) != len(items):
        missing = set(items) - set(item_prices.keys())
        return f"No specials data for: {', '.join(missing)}."

    stores = set.intersection(*[set(prices.keys()) for prices in item_prices.values()])
    for store in stores:
        total = sum(item_prices[item][store][0] for item in items)
        if total <= budget:
            return f"At {store}: {', '.join([f'{item} R{item_prices[item][store][0]}' for item in items])} = R{total}"

    from itertools import product

    best_combo = None
    best_total = float("inf")
    for combo in product(*[prices.items() for prices in item_prices.values()]):
        total = sum(price for _, (price, _, _) in combo)
        if total <= budget and total < best_total:
            best_total = total
            best_combo = combo

    if best_combo:
        return f"Best option: {', '.join([f'{items[i]} R{price} at {store}' for i, (store, (price, _, _)) in enumerate(best_combo)])} = R{best_total}"
    return f"No combination found within R{budget} for {', '.join(items)}."


def listen_for_command():
    try:
        for phrase in LiveSpeech():
            return str(phrase)
    except Exception as e:
        return f"Speech recognition error: {str(e)}"


engine = pyttsx3.init()


def speak_response(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")


def save_image(image_path, description):
    """
    Save an image to the local directory with metadata.

    Args:
        image_path (str): Path to the image file.
        description (str): Description of the image.

    Returns:
        str: Success or error message.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db = get_db("core")
        db["cursor"].execute(
            "INSERT INTO images (path, description, timestamp) VALUES (?, ?, ?)",
            (image_path, description, timestamp),
        )
        db["conn"].commit()
        return f"Image saved successfully with description: '{description}' at {timestamp}."
    except Exception as e:
        return f"Error saving image: {str(e)}"


def read_last_saved_note():
    """
    Retrieve and read the last saved note from the database.

    Returns:
        str: The content of the last saved note or an error message.
    """
    db = get_db("personality")
    db["cursor"].execute("SELECT data FROM research ORDER BY timestamp DESC LIMIT 1")
    result = db["cursor"].fetchone()
    if result:
        note = result[0]
        speak_response(note)
        return f"Last saved note: {note}"
    return "No notes found."

def summarize_last_commands(limit=5):
    """
    Summarize the last few commands issued by the user.

    Args:
        limit (int): Number of commands to summarize.

    Returns:
        str: Summary of the last commands.
    """
    db = get_db("personality")
    db["cursor"].execute("SELECT command FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,))
    results = db["cursor"].fetchall()
    if results:
        summary = "\n".join([f"- {row[0]}" for row in results])
        speak_response(f"Here are your last {limit} commands:\n{summary}")
        return summary
    return "No recent commands found."

def list_all_contacts():
    """
    List all saved contacts from the address book.

    Returns:
        str: A list of all contacts or an error message.
    """
    db = get_db("addressbook")
    db["cursor"].execute("SELECT name, phone, email, category FROM contacts")
    results = db["cursor"].fetchall()
    if results:
        contacts = "\n".join(
            [
                f"Name: {row[0]}, Phone: {row[1] or 'N/A'}, Email: {row[2] or 'N/A'}, Relation: {row[3] or 'N/A'}"
                for row in results
            ]
        )
        speak_response(f"Here are your saved contacts:\n{contacts}")
        return contacts
    return "No contacts found."

def start_wake_word_detection(wake_word_model="resources/wake_word.pmdl", sensitivity=0.5):
    """
    Start listening for the wake word using Snowboy.

    Args:
        wake_word_model (str): Path to the wake word model file.
        sensitivity (float): Sensitivity for wake word detection.

    Returns:
        None
    """
    if not os.path.exists(wake_word_model):
        logging.error(f"Wake word model file not found: {wake_word_model}")
        raise FileNotFoundError(f"Wake word model file not found: {wake_word_model}")

    def detected_callback():
        logging.info("Wake word detected!")
        speak_response("I'm listening.")

    try:
        detector = snowboydecoder.HotwordDetector(wake_word_model, sensitivity=sensitivity)
        logging.info("Listening for wake word...")
        detector.start(detected_callback)
    except Exception as e:
        logging.error(f"Error during wake word detection: {str(e)}")
    finally:
        if 'detector' in locals():
            detector.terminate()
            logging.info("Wake word detection terminated.")

# Ensure Snowboy resources are available
if not os.path.exists("resources/wake_word.pmdl"):
    raise FileNotFoundError("Wake word model file not found in resources folder.")

def list_files_in_directory(directory="."):
    """
    List all files in the specified directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of file names in the directory.
    """
    try:
        files = os.listdir(directory)
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        return f"Error listing files: {str(e)}"

def search_file_by_name(directory=".", filename=""):
    """
    Search for a file by name in the specified directory.

    Args:
        directory (str): Path to the directory.
        filename (str): Name of the file to search for.

    Returns:
        str: Path to the file if found, or an error message.
    """
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
        return f"File '{filename}' not found in directory '{directory}'."
    except Exception as e:
        return f"Error searching for file: {str(e)}"

def copy_file(src, dest):
    """
    Copy a file from source to destination.

    Args:
        src (str): Path to the source file.
        dest (str): Path to the destination directory.

    Returns:
        str: Success or error message.
    """
    try:
        shutil.copy(src, dest)
        return f"File '{src}' copied to '{dest}'."
    except Exception as e:
        return f"Error copying file: {str(e)}"

def add_task_with_deadline(task, deadline):
    """
    Add a task with a deadline to the database.

    Args:
        task (str): The task description.
        deadline (str): The deadline for the task (e.g., '2025-04-10 15:00').

    Returns:
        str: Success or error message.
    """
    try:
        db = get_db("tasks")
        db["cursor"].execute(
            "CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, deadline TEXT, completed INTEGER)"
        )
        db["cursor"].execute(
            "INSERT INTO tasks (task, deadline, completed) VALUES (?, ?, 0)",
            (task, deadline),
        )
        db["conn"].commit()
        return f"Task '{task}' added with deadline {deadline}."
    except Exception as e:
        return f"Error adding task: {str(e)}"

def get_upcoming_tasks():
    """
    Retrieve all upcoming tasks from the database.

    Returns:
        str: List of upcoming tasks or a message if no tasks are found.
    """
    try:
        db = get_db("tasks")
        db["cursor"].execute(
            "SELECT task, deadline FROM tasks WHERE completed = 0 ORDER BY deadline ASC"
        )
        tasks = db["cursor"].fetchall()
        if tasks:
            return "\n".join([f"- {task} (Deadline: {deadline})" for task, deadline in tasks])
        return "No upcoming tasks found."
    except Exception as e:
        return f"Error retrieving tasks: {str(e)}"

def mark_task_as_completed(task_id):
    """
    Mark a task as completed in the database.

    Args:
        task_id (int): The ID of the task to mark as completed.

    Returns:
        str: Success or error message.
    """
    try:
        db = get_db("tasks")
        db["cursor"].execute(
            "UPDATE tasks SET completed = 1 WHERE id = ?",
            (task_id,),
        )
        db["conn"].commit()
        return f"Task ID {task_id} marked as completed."
    except Exception as e:
        return f"Error marking task as completed: {str(e)}"

def remind_upcoming_tasks():
    """
    Use TTS to remind the user of upcoming tasks with deadlines.

    Returns:
        None
    """
    try:
        tasks = get_upcoming_tasks()
        if tasks != "No upcoming tasks found.":
            speak_response(f"Here are your upcoming tasks:\n{tasks}")
        else:
            speak_response("You have no upcoming tasks.")
    except Exception as e:
        speak_response(f"Error reminding tasks: {str(e)}")

def add_note(note, category):
    """
    Add a note with a category to the database.

    Args:
        note (str): The content of the note.
        category (str): The category of the note.

    Returns:
        str: Success or error message.
    """
    try:
        db = get_db("notes")
        db["cursor"].execute(
            "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, note TEXT, category TEXT, timestamp TEXT)"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db["cursor"].execute(
            "INSERT INTO notes (note, category, timestamp) VALUES (?, ?, ?)",
            (note, category, timestamp),
        )
        db["conn"].commit()
        return f"Note added under category '{category}'."
    except Exception as e:
        return f"Error adding note: {str(e)}"

def search_notes_by_category(category):
    """
    Search for notes by category in the database.

    Args:
        category (str): The category to search for.

    Returns:
        str: List of notes in the category or a message if no notes are found.
    """
    try:
        db = get_db("notes")
        db["cursor"].execute(
            "SELECT note, timestamp FROM notes WHERE category = ? ORDER BY timestamp DESC",
            (category,),
        )
        notes = db["cursor"].fetchall()
        if notes:
            return "\n".join([f"- {note} (Added: {timestamp})" for note, timestamp in notes])
        return f"No notes found under category '{category}'."
    except Exception as e:
        return f"Error searching notes: {str(e)}"

def search_notes_by_keyword(keyword):
    """
    Search for notes by keyword in the database.

    Args:
        keyword (str): The keyword to search for.

    Returns:
        str: List of notes containing the keyword or a message if no notes are found.
    """
    try:
        db = get_db("notes")
        db["cursor"].execute(
            "SELECT note, category, timestamp FROM notes WHERE note LIKE ? ORDER BY timestamp DESC",
            (f"%{keyword}%",),
        )
        notes = db["cursor"].fetchall()
        if notes:
            return "\n".join(
                [
                    f"- {note} (Category: {category}, Added: {timestamp})"
                    for note, category, timestamp in notes
                ]
            )
        return f"No notes found containing keyword '{keyword}'."
    except Exception as e:
        return f"Error searching notes: {str(e)}"

def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text or an error message.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def save_extracted_text_as_document(image_path, category):
    """
    Extract text from an image and save it as a note in the database.

    Args:
        image_path (str): Path to the image file.
        category (str): Category under which the extracted text will be saved.

    Returns:
        str: Success or error message.
    """
    try:
        text = extract_text_from_image(image_path)
        if not text:
            return "No text found in the image."
        return add_note(text, category)
    except Exception as e:
        return f"Error saving extracted text: {str(e)}"

def save_command_history(command, response):
    """
    Save a command and its response to the database.

    Args:
        command (str): The user's command.
        response (str): The assistant's response.

    Returns:
        str: Success or error message.
    """
    try:
        db = get_db("history")
        db["cursor"].execute(
            "CREATE TABLE IF NOT EXISTS command_history (id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT, response TEXT, timestamp TEXT)"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db["cursor"].execute(
            "INSERT INTO command_history (command, response, timestamp) VALUES (?, ?, ?)",
            (command, response, timestamp),
        )
        db["conn"].commit()
        return "Command history saved."
    except Exception as e:
        return f"Error saving command history: {str(e)}"

def search_command_history(keyword):
    """
    Search for commands and responses in the history by keyword.

    Args:
        keyword (str): The keyword to search for.

    Returns:
        str: List of matching commands and responses or a message if no matches are found.
    """
    try:
        db = get_db("history")
        db["cursor"].execute(
            "SELECT command, response, timestamp FROM command_history WHERE command LIKE ? OR response LIKE ? ORDER BY timestamp DESC",
            (f"%{keyword}%", f"%{keyword}%"),
        )
        results = db["cursor"].fetchall()
        if results:
            return "\n".join(
                [
                    f"[{timestamp}] Command: {command}\nResponse: {response}"
                    for command, response, timestamp in results
                ]
            )
        return "No matching history found."
    except Exception as e:
        return f"Error searching command history: {str(e)}"

# Function to monitor system resources in real-time
def monitor_resources(interval=1):
    """
    Monitor CPU, memory, and disk usage in real-time.

    Args:
        interval (int): Time interval (in seconds) between updates.

    Returns:
        None
    """
    try:
        while True:
            cpu_usage = psutil.cpu_percent(interval=interval)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory.percent}%")
            print(f"Disk Usage: {disk.percent}%")
            print("-" * 30)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped.")

# Function to list running processes
def list_processes():
    """
    List all running processes and their resource usage.

    Returns:
        list: A list of dictionaries containing process details.
    """
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

# Function to terminate a process by PID
def terminate_process(pid):
    """
    Terminate a process by its PID.

    Args:
        pid (int): Process ID of the process to terminate.

    Returns:
        str: Success or error message.
    """
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        return f"Process {pid} terminated successfully."
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        return f"Error terminating process {pid}: {str(e)}"

# Function to search for files by name or content
def search_files(directory, search_term):
    """
    Search for files by name or content within a directory.

    Args:
        directory (str): Path to the directory.
        search_term (str): Term to search for in file names or content.

    Returns:
        list: List of matching file paths.
    """
    import os

    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if search_term.lower() in file.lower():
                matching_files.append(os.path.join(root, file))
            else:
                try:
                    with open(os.path.join(root, file), 'r', errors='ignore') as f:
                        if search_term.lower() in f.read().lower():
                            matching_files.append(os.path.join(root, file))
                except Exception:
                    pass
    return matching_files

# Function to generate a key for encryption/ decryption
def generate_key():
    """
    Generate a key for encryption and decryption.

    Returns:
        bytes: Encryption key.
    """
    return Fernet.generate_key()

# Function to encrypt a file
def encrypt_file(file_path, key):
    """
    Encrypt a file using the provided key.

    Args:
        file_path (str): Path to the file to encrypt.
        key (bytes): Encryption key.

    Returns:
        str: Success or error message.
    """
    try:
        with open(file_path, 'rb') as file:
            data = file.read()
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        with open(file_path, 'wb') as file:
            file.write(encrypted_data)
        return f"File '{file_path}' encrypted successfully."
    except Exception as e:
        return f"Error encrypting file: {str(e)}"

# Function to decrypt a file
def decrypt_file(file_path, key):
    """
    Decrypt a file using the provided key.

    Args:
        file_path (str): Path to the file to decrypt.
        key (bytes): Encryption key.

    Returns:
        str: Success or error message.
    """
    try:
        with open(file_path, 'rb') as file:
            encrypted_data = file.read()
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data)
        with open(file_path, 'wb') as file:
            file.write(decrypted_data)
        return f"File '{file_path}' decrypted successfully."
    except Exception as e:
        return f"Error decrypting file: {str(e)}"

# Function to create a backup of a file or directory
def create_backup(source_path, backup_path):
    """
    Create a backup of a file or directory.

    Args:
        source_path (str): Path to the file or directory to back up.
        backup_path (str): Path to save the backup.

    Returns:
        str: Success or error message.
    """
    import shutil

    try:
        if os.path.isdir(source_path):
            shutil.copytree(source_path, backup_path)
        else:
            shutil.copy2(source_path, backup_path)
        return f"Backup created at '{backup_path}'."
    except Exception as e:
        return f"Error creating backup: {str(e)}"

# Function to restore a backup
def restore_backup(backup_path, restore_path):
    """
    Restore a backup to the specified location.

    Args:
        backup_path (str): Path to the backup file or directory.
        restore_path (str): Path to restore the backup.

    Returns:
        str: Success or error message.
    """
    import shutil

    try:
        if os.path.isdir(backup_path):
            shutil.copytree(backup_path, restore_path)
        else:
            shutil.copy2(backup_path, restore_path)
        return f"Backup restored to '{restore_path}'."
    except Exception as e:
        return f"Error restoring backup: {str(e)}"

# Function to send email notifications
def send_email_notification(subject, body, recipient_email, sender_email, sender_password):
    """
    Send an email notification.

    Args:
        subject (str): Subject of the email.
        body (str): Body of the email.
        recipient_email (str): Recipient's email address.
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password.

    Returns:
        str: Success or error message.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return "Email sent successfully."
    except Exception as e:
        return f"Error sending email: {str(e)}"

# Function to display desktop notifications
def show_desktop_notification(title, message):
    """
    Display a desktop notification.

    Args:
        title (str): Title of the notification.
        message (str): Message of the notification.

    Returns:
        None
    """
    try:
        toaster = win10toast.ToastNotifier()
        toaster.show_toast(title, message, duration=10)
    except Exception as e:
        print(f"Error displaying notification: {str(e)}")

# Function to send weekly logs via email
def send_weekly_logs(recipient_email, sender_email, sender_password):
    """
    Send weekly logs via email.

    Args:
        recipient_email (str): Recipient's email address.
        sender_email (str): Sender's email address.
        sender_password (str): Sender's email password.

    Returns:
        str: Success or error message.
    """
    try:
        with open('assistant.log', 'r') as log_file:
            logs = log_file.read()

        subject = "Weekly Logs"
        body = f"Here are the weekly logs:\n\n{logs}"

        return send_email_notification(subject, body, recipient_email, sender_email, sender_password)
    except FileNotFoundError:
        return "Log file not found."
    except Exception as e:
        return f"Error sending weekly logs: {str(e)}"

# Function to schedule a task
def schedule_task(task_name, task_function, time_string):
    """
    Schedule a task to run at a specific time.

    Args:
        task_name (str): Name of the task.
        task_function (function): The function to execute.
        time_string (str): Time to run the task (e.g., '14:30').

    Returns:
        str: Success message.
    """
    try:
        schedule.every().day.at(time_string).do(task_function).tag(task_name)
        return f"Task '{task_name}' scheduled at {time_string}."
    except Exception as e:
        return f"Error scheduling task '{task_name}': {str(e)}"

# Function to cancel a scheduled task
def cancel_task(task_name):
    """
    Cancel a scheduled task by its name.

    Args:
        task_name (str): Name of the task to cancel.

    Returns:
        str: Success or error message.
    """
    try:
        schedule.clear(task_name)
        return f"Task '{task_name}' canceled successfully."
    except Exception as e:
        return f"Error canceling task '{task_name}': {str(e)}"

# Function to run the scheduler in a separate thread
def run_scheduler():
    """
    Run the scheduler in a separate thread.

    Returns:
        None
    """
    def scheduler_thread():
        while True:
            schedule.run_pending()
            time.sleep(1)

    threading.Thread(target=scheduler_thread, daemon=True).start()

# Example task function
def example_task():
    print("Example task executed.")

# Function to execute a custom script
def execute_script(script_path):
    """
    Execute a custom script or command.

    Args:
        script_path (str): Path to the script or command to execute.

    Returns:
        str: Success or error message.
    """
    try:
        result = subprocess.run(script_path, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error executing script '{script_path}': {str(e)}"

# Function to scan open ports on a host
def scan_ports(host, start_port=1, end_port=65535):
    """
    Scan open ports on a host.

    Args:
        host (str): Hostname or IP address to scan.
        start_port (int): Starting port number.
        end_port (int): Ending port number.

    Returns:
        list: List of open ports.
    """
    open_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                open_ports.append(port)
    return open_ports

# Function to monitor network traffic
def monitor_network_traffic(interface):
    """
    Monitor incoming and outgoing network traffic.

    Args:
        interface (str): Network interface to monitor.

    Returns:
        None
    """
    try:
        capture = pyshark.LiveCapture(interface=interface)
        print("Monitoring network traffic. Press Ctrl+C to stop.")
        for packet in capture.sniff_continuously():
            print(packet)
    except Exception as e:
        print(f"Error monitoring network traffic: {str(e)}")

# Function to connect to a VPN
def connect_vpn(config_path):
    """
    Connect to a VPN using a configuration file.

    Args:
        config_path (str): Path to the VPN configuration file.

    Returns:
        str: Success or error message.
    """
    try:
        result = subprocess.run(["openvpn", "--config", config_path], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error connecting to VPN: {str(e)}"

# Function to disconnect from a VPN
def disconnect_vpn():
    """
    Disconnect from the VPN.

    Returns:
        str: Success or error message.
    """
    try:
        result = subprocess.run(["taskkill", "/IM", "openvpn.exe", "/F"], capture_output=True, text=True)
        return "VPN disconnected successfully." if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error disconnecting from VPN: {str(e)}"

# Function to access and modify the clipboard
def get_clipboard_content():
    """
    Get the current content of the clipboard.

    Returns:
        str: Clipboard content.
    """
    try:
        return pyperclip.paste()
    except Exception as e:
        return f"Error accessing clipboard: {str(e)}"

def set_clipboard_content(content):
    """
    Set the clipboard content.

    Args:
        content (str): Content to set in the clipboard.

    Returns:
        str: Success message.
    """
    try:
        pyperclip.copy(content)
        return "Clipboard content updated."
    except Exception as e:
        return f"Error setting clipboard content: {str(e)}"

# Function to take a screenshot
def take_screenshot(save_path):
    """
    Capture a screenshot of the entire screen.

    Args:
        save_path (str): Path to save the screenshot.

    Returns:
        str: Success or error message.
    """
    try:
        screenshot = ImageGrab.grab()
        screenshot.save(save_path)
        return f"Screenshot saved at {save_path}."
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"

# Function for power management
def shutdown_system():
    """
    Shutdown the system.

    Returns:
        str: Success or error message.
    """
    try:
        subprocess.run("shutdown /s /t 0", shell=True)
        return "System is shutting down."
    except Exception as e:
        return f"Error shutting down system: {str(e)}"

def restart_system():
    """
    Restart the system.

    Returns:
        str: Success or error message.
    """
    try:
        subprocess.run("shutdown /r /t 0", shell=True)
        return "System is restarting."
    except Exception as e:
        return f"Error restarting system: {str(e)}"

def sleep_system():
    """
    Put the system to sleep.

    Returns:
        str: Success or error message.
    """
    try:
        subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
        return "System is going to sleep."
    except Exception as e:
        return f"Error putting system to sleep: {str(e)}"
