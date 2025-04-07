from datetime import datetime
import os  # Remove this if still unused after fixes  # noqa: F401
import logging
from dotenv import load_dotenv
from config import Config
from apis import (
    get_weather,
    get_news,
    translate_text,
    search_images,
    add_task,
    create_trello_card,
    send_slack_message,
    send_discord_message,
    send_whatsapp_message,
    query_wolfram,
    get_directions,
    get_game_info,
    get_game_news,
    fetch_web_content,
    scrape_store_specials,
)
from offline_tools import (
    listen_for_command,
    speak_response,
    cache_result,
    get_cached_result,
    get_command,
    add_contact,
    get_contact,
    create_database,
    add_to_database,
    get_from_database,
    set_volume,
    open_app,
    type_text,
    create_file,
    delete_file,
    move_file,
    get_cpu_usage,
    get_memory_usage,
    get_disk_space,
    click_at,
    find_and_click,
    log_query,
    save_research,
    get_research,
    analyze_personality,
    feed_data_from_file,
    feed_specials_from_file,
    get_specials,
    find_items_within_budget,
    find_nearest_special,
    get_db,
    databases,  # Added imports
)
from conversation import recognize_intent, generate_response
from coding import generate_code, execute_code, explain_concept, save_command
from calculations import calculate

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler()],
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news,
    "translate_text": translate_text,
    "search_images": search_images,
    "add_task": add_task,
    "create_trello_card": create_trello_card,
    "send_slack_message": send_slack_message,
    "send_discord_message": send_discord_message,
    "send_whatsapp_message": send_whatsapp_message,
    "query_wolfram": query_wolfram,
    "get_directions": get_directions,
    "generate_response": generate_response,
    "generate_code": generate_code,
    "execute_code": execute_code,
    "explain_concept": explain_concept,
    "save_command": save_command,
    "add_contact": add_contact,
    "get_contact": get_contact,
    "create_database": create_database,
    "add_to_database": add_to_database,
    "get_from_database": get_from_database,
    "set_volume": set_volume,
    "open_app": open_app,
    "type_text": type_text,
    "get_game_info": get_game_info,
    "get_game_news": get_game_news,
    "create_file": create_file,
    "delete_file": delete_file,
    "move_file": move_file,
    "get_cpu_usage": get_cpu_usage,
    "get_memory_usage": get_memory_usage,
    "get_disk_space": get_disk_space,
    "click_at": click_at,
    "find_and_click": find_and_click,
    "feed_data_from_file": feed_data_from_file,
    "feed_specials_from_file": feed_specials_from_file,
    "get_specials": get_specials,
    "find_items_within_budget": find_items_within_budget,
    "find_nearest_special": find_nearest_special,
    "scrape_store_specials": lambda store: scrape_and_save_specials(store),
    "research_and_save": lambda topic: research_and_save(topic),
    "get_research": get_research,
    "analyze_personality": analyze_personality,
    "calculate": calculate,
    "complex_if": lambda condition, action: handle_complex_if(condition, action),
    "complex_chain": lambda commands: handle_complex_chain(commands),
}

last_generated_code = {}


def research_and_save(topic):
    result = query_wolfram(topic)
    if "No answer" not in result:
        return save_research(topic, result, "Wolfram Alpha")
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    result = fetch_web_content(url)
    return save_research(topic, result, url)


def scrape_and_save_specials(store):
    specials = scrape_store_specials(store)
    if isinstance(specials, str):  # Error message
        return specials
    db = get_db("specials")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for special in specials:
        db["cursor"].execute(
            "INSERT INTO specials (item, price, store, latitude, longitude, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                special["item"],
                special["price"],
                special["store"],
                special["lat"],
                special["lon"],
                timestamp,
            ),
        )
    db["conn"].commit()
    return f"Specials scraped and saved from {store}."


def handle_complex_if(condition, action):
    if "raining in" in condition:
        city = condition.split("raining in")[-1].strip()
        weather = tools["get_weather"](city=city)
        if "rain" in weather.lower():
            intent, params = recognize_intent(action)
            return tools[intent](**params)
        return "Condition not met, no action taken."
    return "Complex condition not supported yet."


def handle_complex_chain(commands):
    responses = []
    for cmd in commands:
        intent, params = recognize_intent(cmd)
        response = process_single_command(intent, params)
        responses.append(response)
    return "\n".join(responses)


def process_single_command(intent, params):
    if intent in tools:
        try:
            response = tools[intent](**params)
            if intent == "generate_code":
                language = params["language"]
                last_generated_code[language] = response
                response += "\nWould you like me to execute it or save it?"
            elif intent == "execute_code" and not params.get("code"):
                language = params["language"]
                if language in last_generated_code:
                    params["code"] = last_generated_code[language]
                    response = tools[intent](**params)
                else:
                    response = "No code to execute. Please generate some first!"
            elif intent == "save_command":
                if "language" in last_generated_code:
                    language = list(last_generated_code.keys())[0]
                    tools[intent](
                        name=params["name"],
                        language=language,
                        code=last_generated_code[language],
                    )
                    response = f"Command '{params['name']}' saved!"
                else:
                    response = "No code to save yet!"
            return response
        except Exception as e:
            return f"Error processing '{intent}': {str(e)}"
    return "I don’t know how to do that yet!"


def process_command(command):
    global last_generated_code

    cached = get_cached_result(command)
    if cached:
        return f"(Cached) {cached}"

    custom_cmd = get_command(command)
    if custom_cmd:
        language, code = custom_cmd
        return tools["execute_code"](language=language, code=code)

    intent, params = recognize_intent(command)
    topic = (
        params.get("topic") or command.split("about")[-1].strip()
        if "about" in command.lower()
        else None
    )
    if topic:
        log_query(command, topic)
    response = process_single_command(intent, params)
    cache_result(command, response)
    return response


def main():
    logging.info("Starting personal assistant...")
    use_voice = False  # Set to False for testing with input
    while True:
        if use_voice:
            logging.info("Listening...")")
            command = listen_for_command()   command = listen_for_command()
        else:
            command = input("Enter command: ")                command = input("Enter command: ")

        if command.lower() in ["exit", "quit"]:quit"]:
            logging.info("Shutting down...")wn...")
            for db in databases.values():
                db["conn"].close()
            break            logging.info(f"Command: {command}")
and)
        logging.info(f"Command: {command}")nse}")
        response = process_command(command)
        logging.info(f"Response: {response}")_response(response)
        if use_voice:
            speak_response(response)        for db in databases.values():
            db["conn"].close()
urces released. Goodbye!")
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
