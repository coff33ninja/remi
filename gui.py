import pygame
import threading
import queue
from conversation import generate_response
from offline_tools import listen_for_command, speak_response

# Initialize PyGame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FONT_SIZE = 24

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 120, 215)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Assistant GUI")

# Fonts
font = pygame.font.Font(None, FONT_SIZE)

# Input and output queues
input_queue = queue.Queue()
output_queue = queue.Queue()

# Input box
input_box = pygame.Rect(50, SCREEN_HEIGHT - 100, 700, 40)
input_text = ""
input_active = False

# Chat history
chat_history = []

# Function to handle AI responses
def handle_ai_response():
    while True:
        try:
            user_input = input_queue.get()
            if user_input == "voice_command":
                response = listen_for_command()
            else:
                response = generate_response(user_input)
            output_queue.put(response)
        except Exception as e:
            output_queue.put(f"Error: {str(e)}")

# Start AI response handler in a separate thread
threading.Thread(target=handle_ai_response, daemon=True).start()

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Draw chat history
    y_offset = 20
    for message in chat_history[-20:]:  # Show the last 20 messages
        text_surface = font.render(message, True, BLACK)
        screen.blit(text_surface, (50, y_offset))
        y_offset += FONT_SIZE + 10

    # Draw input box
    pygame.draw.rect(screen, GRAY if input_active else BLUE, input_box, 2)
    text_surface = font.render(input_text, True, BLACK)
    screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))

    # Check for AI responses
    try:
        while not output_queue.empty():
            response = output_queue.get()
            chat_history.append(f"AI: {response}")
            speak_response(response)  # Optional: Speak the response
    except Exception as e:
        chat_history.append(f"Error: {str(e)}")

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if input box is clicked
            if input_box.collidepoint(event.pos):
                input_active = True
            else:
                input_active = False
        elif event.type == pygame.KEYDOWN:
            if input_active:
                if event.key == pygame.K_RETURN:
                    chat_history.append(f"You: {input_text}")
                    input_queue.put(input_text)
                    input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
            elif event.key == pygame.K_v:  # Press 'V' to use voice input
                chat_history.append("Listening for voice command...")
                input_queue.put("voice_command")

    # Update display
    pygame.display.flip()

# Quit PyGame
pygame.quit()
