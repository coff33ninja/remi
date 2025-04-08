# gui.py
import pygame
import threading
import queue
import logging
from conversation import generate_response
from offline_tools import listen_for_command, speak_response
import matplotlib.pyplot as plt
import io
import base64
from apis import translate_text
import argparse
import json

# Set up logging to debug response issues
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize PyGame
pygame.init()

# Load theme settings from a configuration file
def load_theme_settings():
    try:
        with open("theme_settings.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "BG_COLOR": [245, 245, 247],
            "TEXT_COLOR": [33, 33, 33],
            "ACCENT_COLOR": [52, 152, 219],
            "SECONDARY_COLOR": [149, 165, 166],
            "CARD_COLOR": [255, 255, 255],
            "FONT_SIZE": 24,
            "CODE_FONT_SIZE": 20,
            "SCREEN_WIDTH": 900,
            "SCREEN_HEIGHT": 700,
        }

# Save theme settings to a configuration file
def save_theme_settings(settings):
    with open("theme_settings.json", "w") as f:
        json.dump(settings, f, indent=4)

# Load initial settings
theme_settings = load_theme_settings()
BG_COLOR = tuple(theme_settings["BG_COLOR"])
TEXT_COLOR = tuple(theme_settings["TEXT_COLOR"])
ACCENT_COLOR = tuple(theme_settings["ACCENT_COLOR"])
SECONDARY_COLOR = tuple(theme_settings["SECONDARY_COLOR"])
CARD_COLOR = tuple(theme_settings["CARD_COLOR"])
FONT_SIZE = theme_settings["FONT_SIZE"]
CODE_FONT_SIZE = theme_settings["CODE_FONT_SIZE"]
SCREEN_WIDTH = theme_settings["SCREEN_WIDTH"]
SCREEN_HEIGHT = theme_settings["SCREEN_HEIGHT"]

# Screen dimensions
FONT_SIZE = 24
CODE_FONT_SIZE = 20

# Modern Colors
BG_COLOR = (245, 245, 247)  # Light gray
TEXT_COLOR = (33, 33, 33)  # Dark gray
ACCENT_COLOR = (52, 152, 219)  # Blue
SECONDARY_COLOR = (149, 165, 166)  # Muted teal
CARD_COLOR = (255, 255, 255)  # White
USER_COLOR = (52, 152, 219)  # Blue for user messages
AI_COLOR = (44, 62, 80)  # Darker gray for AI messages
CODE_BG_COLOR = (230, 230, 230)  # Light gray for code blocks

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("AI Assistant")

# Fonts
font = pygame.font.Font(None, FONT_SIZE)
title_font = pygame.font.Font(None, 36)
code_font = pygame.font.Font(None, CODE_FONT_SIZE)  # Monospaced font for code

# Input and output queues
input_queue = queue.Queue()
output_queue = queue.Queue()

# Input box
input_box = pygame.Rect(50, SCREEN_HEIGHT - 80, 800, 50)
input_text = ""
input_active = False
max_input_lines = 2
placeholder_text = "Type your command..."

# Chat history (limit to 50 messages to prevent overflow)
chat_history = []
chat_scroll_offset = 0  # For scrolling
MAX_CHAT_HISTORY = 50

# Toggle switches
tts_enabled = False
stt_enabled = False
switch_size = 40
tts_switch = pygame.Rect(SCREEN_WIDTH - 200, 40, switch_size * 2, switch_size)
stt_switch = pygame.Rect(SCREEN_WIDTH - 200, 100, switch_size * 2, switch_size)

# Hints section
hints_button = pygame.Rect(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50, 30, 30)
hints_visible = False
hints_rect = pygame.Rect(SCREEN_WIDTH - 350, SCREEN_HEIGHT - 300, 300, 250)
hints_text = [
    "Capabilities:",
    "- Generate code (cmd, ps1, python)",
    "- Execute code snippets",
    "- Explain programming concepts",
    "- Get weather/news",
    "- Translate text",
    "- Search images",
    "- Manage tasks/contacts",
    "- System controls (volume, apps)",
    "- Press 'V' for voice input (STT on, not in text area)",
]


# Function to handle AI responses
def handle_ai_response():
    while True:
        try:
            user_input = input_queue.get()
            logging.info(f"Processing user input: {user_input}")
            if user_input == "voice_command" and stt_enabled:
                response = listen_for_command()
                if response.startswith("Speech recognition error"):
                    response = "Sorry, I couldn't hear you clearly."
            else:
                response = generate_response(user_input)
            logging.info(f"Generated response: {response}")
            output_queue.put(response)
        except Exception as e:
            logging.error(f"Error in handle_ai_response: {str(e)}")
            output_queue.put(f"Error: {str(e)}")


# Start AI response handler in a separate thread
threading.Thread(target=handle_ai_response, daemon=True).start()


# Draw toggle switch
def draw_switch(surface, rect, enabled, label):
    pygame.draw.rect(
        surface,
        SECONDARY_COLOR if not enabled else ACCENT_COLOR,
        rect,
        border_radius=20,
    )
    knob_x = rect.x + (rect.width - switch_size if enabled else 0)
    pygame.draw.circle(
        surface,
        CARD_COLOR,
        (knob_x + switch_size // 2, rect.centery),
        switch_size // 2 - 2,
    )
    label_surface = font.render(label, True, TEXT_COLOR)
    surface.blit(
        label_surface, (rect.x - 150, rect.centery - label_surface.get_height() // 2)
    )


# Draw hints section
def draw_hints(surface):
    if hints_visible:
        pygame.draw.rect(surface, CARD_COLOR, hints_rect, border_radius=10)
        pygame.draw.rect(surface, SECONDARY_COLOR, hints_rect, 1, border_radius=10)
        y_offset = hints_rect.y + 10
        for line in hints_text:
            text_surface = font.render(line, True, TEXT_COLOR)
            surface.blit(text_surface, (hints_rect.x + 10, y_offset))
            y_offset += FONT_SIZE + 5


# Split text into lines based on width
def wrap_text(text, font, max_width, is_code=False):
    if not text:
        return [""]
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if (code_font if is_code else font).size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


# Add a new function to handle the address book form
def draw_address_book_form(surface):
    form_rect = pygame.Rect(100, 100, 700, 500)
    pygame.draw.rect(surface, CARD_COLOR, form_rect, border_radius=10)
    pygame.draw.rect(surface, SECONDARY_COLOR, form_rect, 2, border_radius=10)

    # Form fields
    fields = ["Name", "Surname", "Phone", "Email", "Relation"]
    field_values = {field: "" for field in fields}
    field_rects = {}
    y_offset = form_rect.y + 50

    for field in fields:
        label_surface = font.render(field + ":", True, TEXT_COLOR)
        surface.blit(label_surface, (form_rect.x + 20, y_offset))
        input_rect = pygame.Rect(form_rect.x + 150, y_offset, 500, 40)
        pygame.draw.rect(surface, CARD_COLOR, input_rect, border_radius=5)
        pygame.draw.rect(surface, SECONDARY_COLOR, input_rect, 2, border_radius=5)
        field_rects[field] = input_rect
        y_offset += 60

    # Save button
    save_button = pygame.Rect(form_rect.x + 300, form_rect.y + 400, 100, 40)
    pygame.draw.rect(surface, ACCENT_COLOR, save_button, border_radius=5)
    save_text = font.render("Save", True, CARD_COLOR)
    surface.blit(save_text, save_text.get_rect(center=save_button.center))

    return field_rects, save_button, field_values

# Add event handling for the address book form
def handle_address_book_events(event, field_rects, field_values, save_button):
    global input_active
    if event.type == pygame.MOUSEBUTTONDOWN:
        for field, rect in field_rects.items():
            if rect.collidepoint(event.pos):
                input_active = field
                break
        else:
            input_active = None

        if save_button.collidepoint(event.pos):
            # Save the contact to the database
            add_contact(
                name=field_values["Name"],
                phone=field_values["Phone"],
                email=field_values["Email"],
                address=f"{field_values['Surname']} (Relation: {field_values['Relation']})",
            )
            return "Contact saved successfully!"

    elif event.type == pygame.KEYDOWN and input_active:
        if event.key == pygame.K_BACKSPACE:
            field_values[input_active] = field_values[input_active][:-1]
        else:
            field_values[input_active] += event.unicode

    return None

# Add a toggle for the address book form
address_book_visible = False
address_book_button = pygame.Rect(SCREEN_WIDTH - 300, 40, 200, 40)

def generate_task_chart():
    """
    Generate a bar chart for tasks based on their deadlines.

    Returns:
        str: Base64-encoded image of the chart.
    """
    try:
        db = get_db("tasks")
        db["cursor"].execute(
            "SELECT task, deadline FROM tasks WHERE completed = 0 ORDER BY deadline ASC"
        )
        tasks = db["cursor"].fetchall()

        if not tasks:
            return "No tasks to display."

        task_names = [task[0] for task in tasks]
        deadlines = [task[1] for task in tasks]

        plt.figure(figsize=(10, 6))
        plt.barh(task_names, range(len(task_names)), color="skyblue")
        plt.xlabel("Tasks")
        plt.ylabel("Deadlines")
        plt.title("Task Deadlines")
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        plt.close()

        return image_base64
    except Exception as e:
        return f"Error generating task chart: {str(e)}"

# Add a function to display the chart in the GUI
def display_task_chart(surface):
    """
    Display the task chart in the GUI.

    Args:
        surface: The PyGame surface to draw on.

    Returns:
        None
    """
    chart_data = generate_task_chart()
    if "Error" in chart_data or "No tasks" in chart_data:
        error_surface = font.render(chart_data, True, TEXT_COLOR)
        surface.blit(error_surface, (50, 50))
    else:
        chart_image = pygame.image.load(io.BytesIO(base64.b64decode(chart_data)))
        surface.blit(chart_image, (50, 50))

# Add a dropdown for language selection
language_dropdown = pygame.Rect(SCREEN_WIDTH - 300, 160, 200, 40)
selected_language = "en"  # Default language
available_languages = {"en": "English", "es": "Spanish", "fr": "French", "de": "German"}

def draw_language_dropdown(surface):
    """
    Draw the language selection dropdown in the GUI.

    Args:
        surface: The PyGame surface to draw on.

    Returns:
        None
    """
    pygame.draw.rect(surface, ACCENT_COLOR, language_dropdown, border_radius=5)
    language_text = font.render(
        f"Language: {available_languages[selected_language]}", True, CARD_COLOR
    )
    surface.blit(
        language_text, language_text.get_rect(center=language_dropdown.center)
    )

# Add a settings panel for theme customization
def draw_settings_panel(surface):
    settings_rect = pygame.Rect(100, 100, 700, 500)
    pygame.draw.rect(surface, CARD_COLOR, settings_rect, border_radius=10)
    pygame.draw.rect(surface, SECONDARY_COLOR, settings_rect, 2, border_radius=10)

    y_offset = settings_rect.y + 20
    settings_fields = [
        "BG_COLOR", "TEXT_COLOR", "ACCENT_COLOR", "SECONDARY_COLOR", "CARD_COLOR",
        "FONT_SIZE", "CODE_FONT_SIZE", "SCREEN_WIDTH", "SCREEN_HEIGHT"
    ]
    field_values = {field: str(theme_settings[field]) for field in settings_fields}
    field_rects = {}

    for field in settings_fields:
        label_surface = font.render(field + ":", True, TEXT_COLOR)
        surface.blit(label_surface, (settings_rect.x + 20, y_offset))
        input_rect = pygame.Rect(settings_rect.x + 200, y_offset, 400, 30)
        pygame.draw.rect(surface, CARD_COLOR, input_rect, border_radius=5)
        pygame.draw.rect(surface, SECONDARY_COLOR, input_rect, 2, border_radius=5)
        field_rects[field] = input_rect
        value_surface = font.render(field_values[field], True, TEXT_COLOR)
        surface.blit(value_surface, (input_rect.x + 5, input_rect.y + 5))
        y_offset += 40

    # Save button
    save_button = pygame.Rect(settings_rect.x + 300, settings_rect.y + 450, 100, 40)
    pygame.draw.rect(surface, ACCENT_COLOR, save_button, border_radius=5)
    save_text = font.render("Save", True, CARD_COLOR)
    surface.blit(save_text, save_text.get_rect(center=save_button.center))

    return field_rects, save_button, field_values

# Add event handling for the settings panel
def handle_settings_events(event, field_rects, field_values, save_button):
    global input_active
    if event.type == pygame.MOUSEBUTTONDOWN:
        for field, rect in field_rects.items():
            if rect.collidepoint(event.pos):
                input_active = field
                break
        else:
            input_active = None

        if save_button.collidepoint(event.pos):
            # Save updated settings
            for field in field_values:
                if field in ["FONT_SIZE", "CODE_FONT_SIZE", "SCREEN_WIDTH", "SCREEN_HEIGHT"]:
                    theme_settings[field] = int(field_values[field])
                else:
                    theme_settings[field] = [int(x) for x in field_values[field].strip("[]").split(",")]
            save_theme_settings(theme_settings)
            return "Settings saved successfully!"

    elif event.type == pygame.KEYDOWN and input_active:
        if event.key == pygame.K_BACKSPACE:
            field_values[input_active] = field_values[input_active][:-1]
        else:
            field_values[input_active] += event.unicode

    return None

# Add a toggle for the settings panel
settings_visible = False
settings_button = pygame.Rect(SCREEN_WIDTH - 300, 220, 200, 40)

# Modify the main loop to include a button for displaying the task chart
task_chart_button = pygame.Rect(SCREEN_WIDTH - 300, 100, 200, 40)

def offline_mode_gui():
    """
    Run the GUI in offline mode using local databases and pre-downloaded datasets.

    Returns:
        None
    """
    print("Running GUI in offline mode...")
    # Disable features requiring internet access
    global tts_enabled, stt_enabled
    tts_enabled = False
    stt_enabled = False

    # Main loop for offline mode
    clock = pygame.time.Clock()
    running = True
    while running:
        screen.fill(BG_COLOR)

        # Draw title
        title = title_font.render("AI Assistant (Offline Mode)", True, TEXT_COLOR)
        screen.blit(title, (50, 20))

        # Draw chat area
        chat_rect = pygame.Rect(50, 80, 600, 500)
        pygame.draw.rect(screen, CARD_COLOR, chat_rect, border_radius=10)
        pygame.draw.rect(screen, SECONDARY_COLOR, chat_rect, 1, border_radius=10)

        # Draw chat history
        y_offset = chat_rect.y + 10
        for message in chat_history:
            text_surface = font.render(message, True, TEXT_COLOR)
            screen.blit(text_surface, (chat_rect.x + 10, y_offset))
            y_offset += FONT_SIZE + 5

        # Draw input box
        pygame.draw.rect(screen, CARD_COLOR, input_box, border_radius=10)
        pygame.draw.rect(
            screen,
            ACCENT_COLOR if input_active else SECONDARY_COLOR,
            input_box,
            2,
            border_radius=10,
        )

        # Draw input text or placeholder
        if input_text:
            text_surface = font.render(input_text, True, TEXT_COLOR)
            screen.blit(text_surface, (input_box.x + 10, input_box.y + 5))
        else:
            placeholder_surface = font.render(placeholder_text, True, SECONDARY_COLOR)
            screen.blit(placeholder_surface, (input_box.x + 10, input_box.y + 5))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    input_active = True
                else:
                    input_active = False
            elif event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        if input_text.strip():
                            chat_history.append(f"You: {input_text}")
                            if input_text.lower() == "list contacts":
                                chat_history.append(list_all_contacts())
                            elif input_text.lower() == "list tasks":
                                chat_history.append(get_upcoming_tasks())
                            elif input_text.lower().startswith("add task"):
                                task_details = input_text[9:].strip()
                                task, _, deadline = task_details.partition(" by ")
                                chat_history.append(add_task_with_deadline(task, deadline))
                            elif input_text.lower().startswith("add note"):
                                note_details = input_text[9:].strip()
                                note, _, category = note_details.partition(" in ")
                                chat_history.append(add_note(note, category))
                            else:
                                chat_history.append("Command not recognized in offline mode.")
                            input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Assistant GUI")
    parser.add_argument(
        "--offline", action="store_true", help="Run the GUI in offline mode"
    )
    args = parser.parse_args()

    if args.offline:
        offline_mode_gui()
    else:
        # Main loop
        clock = pygame.time.Clock()
        running = True
        while running:
            screen.fill(BG_COLOR)

            # Draw title
            title = title_font.render("AI Assistant", True, TEXT_COLOR)
            screen.blit(title, (50, 20))

            # Draw chat area
            chat_rect = pygame.Rect(50, 80, 600, 500)
            pygame.draw.rect(screen, CARD_COLOR, chat_rect, border_radius=10)
            pygame.draw.rect(screen, SECONDARY_COLOR, chat_rect, 1, border_radius=10)

            # Draw chat history with scrolling
            y_offset = chat_rect.y + 10
            total_height = 0
            for message in chat_history:
                is_user = message.startswith("You:")
                is_code = "\n" in message and "AI:" in message
                prefix = "You: " if is_user else "AI: "
                text = message[5:] if is_user else message[4:]
                color = USER_COLOR if is_user else AI_COLOR
                font_to_use = code_font if is_code else font

                # Render prefix
                prefix_surface = font.render(prefix, True, color)
                prefix_width = prefix_surface.get_width()

                # Wrap the message content
                wrapped_lines = wrap_text(
                    text, font_to_use, chat_rect.width - 20 - prefix_width, is_code
                )

                # Calculate total height of message
                line_height = CODE_FONT_SIZE + 5 if is_code else FONT_SIZE + 5
                message_height = len(wrapped_lines) * line_height
                total_height += message_height

                # Adjust y_offset with scroll offset
                adjusted_y = y_offset + chat_scroll_offset
                if (
                    adjusted_y + message_height >= chat_rect.y
                    and adjusted_y <= chat_rect.bottom
                ):
                    for i, line in enumerate(wrapped_lines):
                        text_surface = font_to_use.render(line, True, color)
                        x_pos = (
                            chat_rect.x + 10
                            if is_user
                            else chat_rect.x + chat_rect.width - text_surface.get_width() - 10
                        )
                        if is_user:
                            if i == 0:
                                screen.blit(prefix_surface, (chat_rect.x + 10, adjusted_y))
                                screen.blit(
                                    text_surface, (chat_rect.x + 10 + prefix_width, adjusted_y)
                                )
                            else:
                                screen.blit(
                                    text_surface,
                                    (
                                        chat_rect.x + 10 + prefix_width,
                                        adjusted_y + i * line_height,
                                    ),
                                )
                        else:
                            if i == 0:
                                screen.blit(prefix_surface, (chat_rect.x + 10, adjusted_y))
                                screen.blit(
                                    text_surface, (chat_rect.x + 10 + prefix_width, adjusted_y)
                                )
                            else:
                                screen.blit(
                                    text_surface,
                                    (
                                        chat_rect.x + 10 + prefix_width,
                                        adjusted_y + i * line_height,
                                    ),
                                )
                            if is_code:
                                pygame.draw.rect(
                                    screen,
                                    CODE_BG_COLOR,
                                    (
                                        chat_rect.x + 10 + prefix_width - 5,
                                        adjusted_y + i * line_height - 2,
                                        text_surface.get_width() + 10,
                                        line_height,
                                    ),
                                    border_radius=5,
                                )
                y_offset += message_height

            # Auto-scroll to the bottom by default
            if total_height > chat_rect.height:
                chat_scroll_offset = -(total_height - chat_rect.height + 20)
            else:
                chat_scroll_offset = 0

            # Draw input box
            pygame.draw.rect(screen, CARD_COLOR, input_box, border_radius=10)
            pygame.draw.rect(
                screen,
                ACCENT_COLOR if input_active else SECONDARY_COLOR,
                input_box,
                2,
                border_radius=10,
            )

            # Draw input text or placeholder
            if input_text:
                wrapped_lines = wrap_text(input_text, font, input_box.width - 20)
                for i, line in enumerate(wrapped_lines[:max_input_lines]):
                    text_surface = font.render(line, True, TEXT_COLOR)
                    screen.blit(
                        text_surface, (input_box.x + 10, input_box.y + 5 + i * (FONT_SIZE + 5))
                    )
            else:
                placeholder_surface = font.render(placeholder_text, True, SECONDARY_COLOR)
                screen.blit(placeholder_surface, (input_box.x + 10, input_box.y + 5))

            # Draw toggle switches
            draw_switch(screen, tts_switch, tts_enabled, "Text-to-Speech")
            draw_switch(screen, stt_switch, stt_enabled, "Speech-to-Text")

            # Draw hints button
            pygame.draw.circle(screen, ACCENT_COLOR, hints_button.center, 15)
            question_mark = font.render("?", True, CARD_COLOR)
            screen.blit(question_mark, question_mark.get_rect(center=hints_button.center))

            # Draw hints if visible
            draw_hints(screen)

            # Draw address book button
            pygame.draw.rect(screen, ACCENT_COLOR, address_book_button, border_radius=5)
            address_book_text = font.render("Address Book", True, CARD_COLOR)
            screen.blit(
                address_book_text, address_book_text.get_rect(center=address_book_button.center)
            )

            if address_book_visible:
                field_rects, save_button, field_values = draw_address_book_form(screen)

            # Draw task chart button
            pygame.draw.rect(screen, ACCENT_COLOR, task_chart_button, border_radius=5)
            task_chart_text = font.render("Task Chart", True, CARD_COLOR)
            screen.blit(
                task_chart_text, task_chart_text.get_rect(center=task_chart_button.center)
            )

            # Draw language dropdown
            draw_language_dropdown(screen)

            # Draw settings button
            pygame.draw.rect(screen, ACCENT_COLOR, settings_button, border_radius=5)
            settings_text = font.render("Settings", True, CARD_COLOR)
            screen.blit(settings_text, settings_text.get_rect(center=settings_button.center))

            if settings_visible:
                field_rects, save_button, field_values = draw_settings_panel(screen)

            # Check for AI responses
            try:
                while not output_queue.empty():
                    response = output_queue.get()
                    logging.info(f"Received response from queue: {response}")
                    # Translate response to the selected language
                    translated_response = translate_text(response, selected_language)
                    chat_history.append(f"AI: {translated_response}")
                    if len(chat_history) > MAX_CHAT_HISTORY:
                        chat_history.pop(0)
                    if tts_enabled:
                        speak_response(translated_response)
            except Exception as e:
                logging.error(f"Error in output queue: {str(e)}")
                chat_history.append(f"Error: {str(e)}")
                if len(chat_history) > MAX_CHAT_HISTORY:
                    chat_history.pop(0)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if input_box.collidepoint(event.pos):
                        input_active = True
                        hints_visible = False
                    elif hints_button.collidepoint(event.pos):
                        hints_visible = not hints_visible
                    elif tts_switch.collidepoint(event.pos):
                        tts_enabled = not tts_enabled
                    elif stt_switch.collidepoint(event.pos):
                        stt_enabled = not stt_enabled
                    elif address_book_button.collidepoint(event.pos):
                        address_book_visible = not address_book_visible
                    elif address_book_visible:
                        message = handle_address_book_events(event, field_rects, field_values, save_button)
                        if message:
                            chat_history.append(f"AI: {message}")
                    elif task_chart_button.collidepoint(event.pos):
                        display_task_chart(screen)
                    elif language_dropdown.collidepoint(event.pos):
                        # Cycle through available languages
                        lang_keys = list(available_languages.keys())
                        current_index = lang_keys.index(selected_language)
                        selected_language = lang_keys[(current_index + 1) % len(lang_keys)]
                    elif settings_button.collidepoint(event.pos):
                        settings_visible = not settings_visible
                    elif settings_visible:
                        message = handle_settings_events(event, field_rects, field_values, save_button)
                        if message:
                            chat_history.append(f"AI: {message}")
                    else:
                        input_active = False
                        hints_visible = False
                elif event.type == pygame.KEYDOWN:
                    if input_active:
                        if event.key == pygame.K_RETURN:
                            if input_text.strip():
                                chat_history.append(f"You: {input_text}")
                                if len(chat_history) > MAX_CHAT_HISTORY:
                                    chat_history.pop(0)
                                input_queue.put(input_text)
                                input_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            if input_text:
                                input_text = input_text[:-1]
                        else:
                            test_text = input_text + event.unicode
                            if (
                                len(wrap_text(test_text, font, input_box.width - 20))
                                <= max_input_lines
                            ):
                                input_text = test_text
                    elif event.key == pygame.K_v and stt_enabled and not input_active:
                        chat_history.append("Listening for voice command...")
                        if len(chat_history) > MAX_CHAT_HISTORY:
                            chat_history.pop(0)
                        input_queue.put("voice_command")
                    elif event.key == pygame.K_t:  # Toggle TTS
                        tts_enabled = not tts_enabled
                    elif event.key == pygame.K_s:  # Toggle STT
                        stt_enabled = not stt_enabled
                elif event.type == pygame.MOUSEWHEEL:
                    chat_scroll_offset += event.y * 20
                    chat_scroll_offset = min(chat_scroll_offset, 0)
                    if total_height > chat_rect.height:
                        max_offset = -(total_height - chat_rect.height + 20)
                        chat_scroll_offset = max(chat_scroll_offset, max_offset)

            # Update display
            pygame.display.flip()
            clock.tick(60)

        # Quit PyGame
        pygame.quit()
