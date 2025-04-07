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

nlp = spacy.load("en_core_web_sm")

databases = {}


def get_db(db_name):
    if db_name not in databases:
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
