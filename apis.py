import httpx  # Replace requests with httpx for async support
import requests  # Added missing import
import os
from dotenv import load_dotenv  # Add this import
import pywhatkit
from bs4 import BeautifulSoup
import logging
import time  # Add this import for retries

# Load environment variables from .env file
load_dotenv()  # Add this line before accessing env vars

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
    "google_calendar": os.getenv("GOOGLE_CALENDAR_API_KEY"),
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "coingecko": os.getenv("COINGECKO_API_KEY"),
    "tmdb": os.getenv("TMDB_API_KEY"),
    "nutritionix": os.getenv("NUTRITIONIX_API_KEY"),
    "nutritionix_app_id": os.getenv("NUTRITIONIX_APP_ID"),
    "spotify": os.getenv("SPOTIFY_API_KEY"),
    "skyscanner": os.getenv("SKYSCANNER_API_KEY"),
    "amadeus": os.getenv("AMADEUS_API_KEY"),
}

def validate_api_keys():
    missing_keys = [key for key, value in API_KEYS.items() if not value]
    if missing_keys:
        logging.error(f"Missing API keys: {', '.join(missing_keys)}")
        raise EnvironmentError(f"Missing API keys: {', '.join(missing_keys)}")

validate_api_keys()

def retry_request(url, retries=3, backoff_factor=2):
    for attempt in range(retries):
        try:
            response = httpx.get(url, timeout=10)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            if attempt < retries - 1:
                time.sleep(backoff_factor ** attempt)
            else:
                raise e

def get_weather(city, api_key=API_KEYS["openweathermap"]):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = retry_request(url)
        data = response.json()
        if "weather" in data:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The weather in {city} is {desc} with a temperature of {temp}°C."
        return "Could not retrieve weather data."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

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
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            return data["results"][0]["urls"]["regular"]
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

def create_trello_card(title, desc, api_key=API_KEYS["trello_key"], token=API_KEYS["trello_token"]):
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
            name = item.select_one(".item-name").text
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

def add_google_calendar_event(summary, start_time, end_time, description=None):
    """
    Add an event to Google Calendar.

    Args:
        summary (str): Title of the event.
        start_time (str): Start time in ISO 8601 format (e.g., '2025-04-10T15:00:00Z').
        end_time (str): End time in ISO 8601 format (e.g., '2025-04-10T16:00:00Z').
        description (str, optional): Description of the event.

    Returns:
        str: Success or error message.
    """
    try:
        url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        headers = {
            "Authorization": f"Bearer {API_KEYS['google_calendar']}",
            "Content-Type": "application/json",
        }
        event_data = {
            "summary": summary,
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }
        if description:
            event_data["description"] = description

        response = requests.post(url, headers=headers, json=event_data)
        response.raise_for_status()
        return f"Event '{summary}' added to Google Calendar."
    except requests.exceptions.RequestException as e:
        return f"Error adding event to Google Calendar: {str(e)}"

def get_upcoming_google_calendar_events():
    """
    Retrieve upcoming events from Google Calendar.

    Returns:
        str: List of upcoming events or an error message.
    """
    try:
        url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        headers = {
            "Authorization": f"Bearer {API_KEYS['google_calendar']}",
        }
        params = {"timeMin": datetime.now().isoformat() + "Z", "maxResults": 10, "singleEvents": True, "orderBy": "startTime"}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        events = response.json().get("items", [])

        if not events:
            return "No upcoming events found."

        event_list = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            event_list.append(f"- {event['summary']} (Start: {start})")

        return "\n".join(event_list)
    except requests.exceptions.RequestException as e:
        return f"Error retrieving events from Google Calendar: {str(e)}"

# Finance API: Alpha Vantage
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
def get_stock_price(symbol, api_key=API_KEYS.get("alpha_vantage")):
    """
    Fetch the stock price for a given symbol using Alpha Vantage.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL" for Apple).
        api_key (str): API key for Alpha Vantage.

    Returns:
        str: Stock price or an error message.
    """
    try:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "1min",
            "apikey": api_key,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        time_series = data.get("Time Series (1min)")
        if time_series:
            latest_time = sorted(time_series.keys())[0]
            price = time_series[latest_time]["1. open"]
            return f"The latest price for {symbol} is ${price}."
        return "Stock data not available."
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"

# Finance API: CoinGecko
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/simple/price"
def get_crypto_price(crypto, currency="usd"):
    """
    Fetch the cryptocurrency price using CoinGecko.

    Args:
        crypto (str): Cryptocurrency symbol (e.g., "bitcoin").
        currency (str): Currency to convert to (default: "usd").

    Returns:
        str: Cryptocurrency price or an error message.
    """
    try:
        params = {"ids": crypto, "vs_currencies": currency}
        response = requests.get(COINGECKO_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        price = data.get(crypto, {}).get(currency)
        if price:
            return f"The current price of {crypto} is {price} {currency.upper()}."
        return "Cryptocurrency data not available."
    except Exception as e:
        return f"Error fetching cryptocurrency price: {str(e)}"

# Entertainment API: TMDb
TMDB_BASE_URL = "https://api.themoviedb.org/3/search/movie"
def search_movie(movie_name, api_key=API_KEYS.get("tmdb")):
    """
    Search for a movie using TMDb.

    Args:
        movie_name (str): Name of the movie to search for.
        api_key (str): API key for TMDb.

    Returns:
        str: Movie details or an error message.
    """
    try:
        params = {"query": movie_name, "api_key": api_key}
        response = requests.get(TMDB_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results:
            movie = results[0]
            return f"{movie['title']} ({movie['release_date']}): {movie['overview']}"
        return "Movie not found."
    except Exception as e:
        return f"Error searching for movie: {str(e)}"

# Entertainment API: Spotify (Placeholder for OAuth setup)
def get_spotify_recommendations():
    """
    Fetch music recommendations from Spotify.

    Returns:
        str: Recommendations or a placeholder message.
    """
    return "Spotify integration is under development."

# Health and Fitness API: Nutritionix
NUTRITIONIX_BASE_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"
def get_nutritional_info(food_item, api_key=API_KEYS.get("nutritionix")):
    """
    Fetch nutritional information for a food item using Nutritionix.

    Args:
        food_item (str): Name of the food item.
        api_key (str): API key for Nutritionix.

    Returns:
        str: Nutritional information or an error message.
    """
    try:
        headers = {
            "x-app-id": API_KEYS.get("nutritionix_app_id"),
            "x-app-key": api_key,
        }
        data = {"query": food_item}
        response = requests.post(NUTRITIONIX_BASE_URL, headers=headers, json=data)
        response.raise_for_status()
        data = response.json()
        if data.get("foods"):
            food = data["foods"][0]
            return f"{food['food_name']}: {food['nf_calories']} calories, {food['nf_protein']}g protein."
        return "Nutritional information not available."
    except Exception as e:
        return f"Error fetching nutritional information: {str(e)}"

# Travel API: Skyscanner (Placeholder for API setup)
def search_flights():
    """
    Search for flights using Skyscanner.

    Returns:
        str: Flight details or a placeholder message.
    """
    return "Skyscanner integration is under development."

# Travel API: Amadeus (Placeholder for API setup)
def search_hotels():
    """
    Search for hotels using Amadeus.

    Returns:
        str: Hotel details or a placeholder message.
    """
    return "Amadeus integration is under development."
