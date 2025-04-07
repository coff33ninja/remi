import httpx  # Replace requests with httpx for async support
import requests  # Added missing import
import os
import pywhatkit
from bs4 import BeautifulSoup
import logging

API_KEYS = {
    "openweathermap": os.getenv("OPENWEATHERMAP_API_KEY"),
    "newsapi": os.getenv("NEWSAPI_API_KEY"),
    "deepl": os.getenv("DEEPL_API_KEY"),
    "unsplash": os.getenv("UNSPLASH_API_KEY"),
    "todoist": os.getenv("TODOIST_API_KEY"),
    "trello": os.getenv("TRELLO_API_KEY"),
    "trello_key": os.getenv("TRELLO_API_KEY"),
    "trello_token": os.getenv("TRELLO_TOKEN"),
    "slack": os.getenv("SLACK_API_TOKEN"),
    "discord": os.getenv("DISCORD_BOT_TOKEN"),
    "wolframalpha": os.getenv("WOLFRAM_ALPHA_APP_ID"),
    "googlemaps": os.getenv("GOOGLE_MAPS_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_TOKEN"),
}

def validate_api_keys():
    missing_keys = [key for key, value in API_KEYS.items() if not value]
    if missing_keys:
        logging.warning(f"Missing API keys: {', '.join(missing_keys)}")
        # Uncomment the next line to enforce validation
        # raise EnvironmentError(f"Missing API keys: {', '.join(missing_keys)}")

validate_api_keys()

async def get_weather(city, api_key=API_KEYS["openweathermap"]):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
        if data.get("weather"):
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The weather in {city} is {desc} with a temperature of {temp}°C."
        return "Could not retrieve weather data."
    except Exception as e:
        return f"Weather API error: {str(e)}"

def get_news(topic, api_key=API_KEYS["newsapi"]):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        articles = response.json().get("articles", [])[:3]
        if articles:
            return "\n".join(
                [f"{i+1}. {article['title']}" for i, article in enumerate(articles)]
            )
        return "No news found."
    except requests.exceptions.RequestException as e:
        return f"News API error: {str(e)}"

def translate_text(text, target_lang, api_key=API_KEYS["deepl"]):
    url = "https://api-free.deepl.com/v2/translate"
    params = {"auth_key": api_key, "text": text, "target_lang": target_lang.upper()}
    try:
        response = requests.post(url, data=params).json()
        return response["translations"][0]["text"]
    except Exception as e:
        return f"Translation error: {str(e)}"

def search_images(query, api_key=API_KEYS["unsplash"]):
    url = f"https://api.unsplash.com/search/photos?query={query}&client_id={api_key}"
    try:
        response = requests.get(url).json()
        if response.get("results"):
            return response["results"][0]["urls"]["small"]
        return "No images found."
    except Exception as e:
        return f"Image search error: {str(e)}"

def add_task(task, due=None, api_key=API_KEYS["todoist"]):
    url = "https://api.todoist.com/rest/v2/tasks"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"content": task}
    if due:
        data["due_string"] = due
    try:
        response = requests.post(url, headers=headers, json=data).json()
        return f"Task '{task}' added with ID: {response['id']}"
    except Exception as e:
        return f"Todoist error: {str(e)}"

def create_trello_card(
    title, desc, api_key=API_KEYS["trello_key"], token=API_KEYS["trello_token"]
):
    url = "https://api.trello.com/1/cards"
    params = {
        "key": api_key,
        "token": token,
        "idList": "your_list_id_here",
        "name": title,
        "desc": desc,
    }
    try:
        response = requests.post(url, params=params).json()
        return f"Trello card '{title}' created."
    except Exception as e:
        return f"Trello error: {str(e)}"

def send_slack_message(channel, text, api_key=API_KEYS["slack"]):
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"channel": channel, "text": text}
    try:
        response = requests.post(url, headers=headers, json=data).json()
        return "Message sent to Slack." if response["ok"] else response["error"]
    except Exception as e:
        return f"Slack error: {str(e)}"

def send_discord_message(channel_id, text, api_key=API_KEYS["discord"]):
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {api_key}"}
    data = {"content": text}
    try:
        response = requests.post(url, headers=headers, json=data).json()
        if response.get("id"):
            return "Message sent to Discord."
        return response.get("message", "Error")
    except Exception as e:
        return f"Discord error: {str(e)}"

def send_whatsapp_message(phone, message):
    try:
        pywhatkit.sendwhatmsg_instantly(phone, message, wait_time=10, tab_close=True)
        return "WhatsApp message sent!"
    except Exception as e:
        return f"WhatsApp error: {str(e)}"

def query_wolfram(query, api_key=API_KEYS["wolframalpha"]):
    url = f"http://api.wolframalpha.com/v2/query?input={query}&appid={api_key}&output=json"
    try:
        response = requests.get(url).json()
        pods = response["queryresult"]["pods"]
        if pods:
            return pods[0]["subpods"][0]["plaintext"]
        return "No answer from Wolfram Alpha."
    except Exception as e:
        return f"Wolfram Alpha error: {str(e)}"

def get_directions(origin, destination, api_key=API_KEYS["googlemaps"]):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={api_key}"
    try:
        response = requests.get(url).json()
        if response["routes"]:
            distance = response["routes"][0]["legs"][0]["distance"]["text"]
            return f"Distance from {origin} to {destination}: {distance}"
        return "Could not find directions."
    except Exception as e:
        return f"Google Maps error: {str(e)}"

def get_game_info(game):
    url = f"https://en.wikipedia.org/wiki/{game.replace(' ', '_')}"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        intro = soup.find("p", class_=None).text.strip()
        return intro[:200] + "..." if len(intro) > 200 else intro
    except Exception as e:
        return f"Game info error: {str(e)}"

def get_game_news(game, api_key=API_KEYS["newsapi"]):
    return get_news(f"{game} gaming", api_key)

def fetch_web_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(p.text for p in soup.find_all("p"))
        return text[:500] + "..." if len(text) > 500 else text
    except Exception as e:
        return f"Web fetch error: {str(e)}"

def scrape_store_specials(store):
    store_urls = {
        "shoprite": "https://example.com/shoprite-specials",
        "picknpay": "https://example.com/picknpay-specials",
        "checkers": "https://example.com/checkers-specials",
    }
    store_locations = {
        "shoprite": {"lat": -33.9249, "lon": 18.4241},  # Cape Town example
        "picknpay": {"lat": -33.9321, "lon": 18.4523},
        "checkers": {"lat": -33.9180, "lon": 18.4380},
    }
    url = store_urls.get(store.lower(), "")
    if not url:
        return f"No URL configured for {store}."

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        specials = []
        for item in soup.select(".special-item"):  # Hypothetical class
            name = item.select_one(".item-name").text.lower()
            price = float(item.select_one(".item-price").text.replace("R", ""))
            specials.append(
                {
                    "item": name,
                    "price": price,
                    "store": store.capitalize(),
                    "lat": store_locations[store.lower()]["lat"],
                    "lon": store_locations[store.lower()]["lon"],
                }
            )
        return specials
    except Exception as e:
        return f"Web scraping error for {store}: {str(e)}"
