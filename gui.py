import pygame
import threading
import queue
from conversation import generate_response
from offline_tools import listen_for_command, speak_response

# Initialize PyGame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 700
FONT_SIZE = 24

# Modern Colors
BG_COLOR = (245, 245, 247)  # Light gray
TEXT_COLOR = (33, 33, 33)  # Dark gray
ACCENT_COLOR = (52, 152, 219)  # Blue
SECONDARY_COLOR = (149, 165, 166)  # Muted teal
CARD_COLOR = (255, 255, 255)  # White

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Assistant")

# Fonts
font = pygame.font.Font(None, FONT_SIZE)
title_font = pygame.font.Font(None, 36)

# Input and output queues
input_queue = queue.Queue()
output_queue = queue.Queue()

# Input box
input_box = pygame.Rect(50, SCREEN_HEIGHT - 80, 800, 50)
input_text = ""
input_active = False
max_input_lines = 2  # Limit to 2 lines for simplicity

# Chat history
chat_history = []

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
            if user_input == "voice_command" and stt_enabled:
                response = listen_for_command()
                if response.startswith("Speech recognition error"):
                    response = "Sorry, I couldn't hear you clearly."
            else:
                response = generate_response(user_input)
            output_queue.put(response)
        except Exception as e:
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
def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines[:max_input_lines]  # Limit to max lines


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

    # Draw chat history
    y_offset = 90
    for message in chat_history[-15:]:
        if y_offset + FONT_SIZE + 10 < chat_rect.bottom:
            text_surface = font.render(message, True, TEXT_COLOR)
            text_rect = text_surface.get_rect(topleft=(chat_rect.x + 10, y_offset))
            pygame.draw.rect(
                screen, CARD_COLOR, text_rect.inflate(10, 5), border_radius=5
            )
            screen.blit(text_surface, text_rect.topleft)
            y_offset += FONT_SIZE + 15

    # Draw input box
    pygame.draw.rect(screen, CARD_COLOR, input_box, border_radius=10)
    pygame.draw.rect(
        screen,
        ACCENT_COLOR if input_active else SECONDARY_COLOR,
        input_box,
        2,
        border_radius=10,
    )

    # Wrap and render input text
    wrapped_lines = wrap_text(input_text, font, input_box.width - 20)  # 20 for padding
    for i, line in enumerate(wrapped_lines):
        text_surface = font.render(line, True, TEXT_COLOR)
        screen.blit(
            text_surface, (input_box.x + 10, input_box.y + 5 + i * (FONT_SIZE + 5))
        )

    # Draw toggle switches
    draw_switch(screen, tts_switch, tts_enabled, "Text-to-Speech")
    draw_switch(screen, stt_switch, stt_enabled, "Speech-to-Text")

    # Draw hints button (question mark)
    pygame.draw.circle(screen, ACCENT_COLOR, hints_button.center, 15)
    question_mark = font.render("?", True, CARD_COLOR)
    screen.blit(question_mark, question_mark.get_rect(center=hints_button.center))

    # Draw hints if visible
    draw_hints(screen)

    # Check for AI responses
    try:
        while not output_queue.empty():
            response = output_queue.get()
            chat_history.append(f"AI: {response}")
            if tts_enabled:
                speak_response(response)
    except Exception as e:
        chat_history.append(f"Error: {str(e)}")

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
            else:
                input_active = False
                hints_visible = False
        elif event.type == pygame.KEYDOWN:
            if input_active:
                if event.key == pygame.K_RETURN:
                    if input_text.strip():
                        chat_history.append(f"You: {input_text}")
                        input_queue.put(input_text)
                        input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    if input_text:
                        input_text = input_text[:-1]
                else:
                    # Only add character if it fits within max lines
                    test_text = input_text + event.unicode
                    if (
                        len(wrap_text(test_text, font, input_box.width - 20))
                        <= max_input_lines
                    ):
                        input_text = test_text
            elif event.key == pygame.K_v and stt_enabled and not input_active:
                chat_history.append("Listening for voice command...")
                input_queue.put("voice_command")

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit PyGame
pygame.quit()
