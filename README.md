# Personal Assistant - REMI

REMI is a personal assistant project designed to handle various tasks such as fetching weather updates, managing tasks, generating code, and more. This README provides an in-depth guide to setting up and using REMI.

---

## Table of Contents

1. [Features](#features)
2. [Setup Instructions](#setup-instructions)
3. [Environment Variables and API Keys](#environment-variables-and-api-keys)
4. [Supported Commands](#supported-commands)
5. [Data Placement](#data-placement)
6. [System Requirements](#system-requirements)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)

---

## Features

- Fetch weather updates
- Retrieve news articles
- Translate text
- Search for images
- Manage tasks (Todoist, Trello)
- Send messages (Slack, Discord, WhatsApp)
- Query Wolfram Alpha
- Get directions (Google Maps)
- Generate and execute code
- Manage files and databases
- Analyze personality based on queries
- Web scraping for store specials
- Advanced model switching for code generation

---

## Setup Instructions

### 1. Install Dependencies

Run the following commands to install the required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you have a GPU and want to leverage it, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Ensure Tesseract OCR is installed for PDF parsing.

### 2. Clone the Repository

```bash
git clone https://github.com/coff33ninja/remi.git
cd remi
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory and add the following:

```env
DEBUG=true
API_TIMEOUT=10
LOG_FILE=assistant.log

# API Keys
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
NEWSAPI_API_KEY=your_newsapi_api_key
DEEPL_API_KEY=your_deepl_api_key
UNSPLASH_API_KEY=your_unsplash_api_key
TODOIST_API_KEY=your_todoist_api_key
TRELLO_API_KEY=your_trello_api_key
TRELLO_TOKEN=your_trello_token
SLACK_API_TOKEN=your_slack_api_token
DISCORD_BOT_TOKEN=your_discord_bot_token
WOLFRAM_ALPHA_APP_ID=your_wolfram_alpha_app_id
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

---

## Environment Variables and API Keys

### Required API Keys

| Service            | API Key Name               | Purpose                          | Link to Obtain API Key |
|--------------------|----------------------------|----------------------------------|------------------------|
| OpenWeatherMap     | `OPENWEATHERMAP_API_KEY`   | Fetch weather updates            | [OpenWeatherMap](https://home.openweathermap.org/users/sign_up) |
| NewsAPI            | `NEWSAPI_API_KEY`          | Retrieve news articles           | [NewsAPI](https://newsapi.org/register) |
| DeepL              | `DEEPL_API_KEY`            | Translate text                   | [DeepL](https://www.deepl.com/pro#developer) |
| Unsplash           | `UNSPLASH_API_KEY`         | Search for images                | [Unsplash](https://unsplash.com/developers) |
| Todoist            | `TODOIST_API_KEY`          | Manage tasks                     | [Todoist](https://developer.todoist.com/appconsole.html) |
| Trello             | `TRELLO_API_KEY`, `TRELLO_TOKEN` | Manage Trello cards      | [Trello](https://trello.com/app-key) |
| Slack              | `SLACK_API_TOKEN`          | Send Slack messages              | [Slack](https://api.slack.com/apps) |
| Discord            | `DISCORD_BOT_TOKEN`        | Send Discord messages            | [Discord](https://discord.com/developers/applications) |
| Wolfram Alpha      | `WOLFRAM_ALPHA_APP_ID`     | Query Wolfram Alpha              | [Wolfram Alpha](https://developer.wolframalpha.com/portal/myapps/) |
| Google Maps        | `GOOGLE_MAPS_API_KEY`      | Get directions                   | [Google Maps](https://developers.google.com/maps/documentation/javascript/get-api-key) |

---

## Supported Commands

### General Commands

- **Weather**: `What's the weather in [city]?`
- **News**: `Get news about [topic].`
- **Translate**: `Translate [text] to [language].`
- **Images**: `Search for images of [query].`
- **Tasks**: `Add task [task] due [date].`
- **Slack**: `Send [message] to Slack.`
- **Discord**: `Send [message] to Discord.`
- **WhatsApp**: `Send [message] to [phone number] on WhatsApp.`
- **Directions**: `Get directions from [origin] to [destination].`

### Code Generation

- **Generate Code**: `Write Python code to [task].`
- **Execute Code**: `Run Python code [code].`
- **Explain Concept**: `Explain [concept] in Python.`

### File and Database Management

- **Create File**: `Create file [filename].`
- **Delete File**: `Delete file [filename].`
- **Move File**: `Move file [source] to [destination].`
- **Database**: `Create database [name].`

### Advanced Features

- **Switch Models**: Automatically switch to a better model if the smaller one fails.
- **Analyze Personality**: `Tell me about myself.`
- **Web Scraping**: `Scrape specials from [store].`

---

## Data Placement

### Specials File

- **Text File**: Place in the format `item:price:store:lat:lon` (e.g., `milk:15.99:Shoprite:-33.9249:18.4241`).
- **PDF File**: Text like `milk R15.99 at Shoprite lat -33.9249 lon 18.4241`.

### Database Files

- SQLite databases are automatically created in the project directory.

### Steps to train the model
1. Install the required packages if not done so prior: `pip install peft transformers torch'
2. Run the training script: `python train_lora.py`
3. The model will be saved in the `./fine_tuned_model` directory.

---

## System Requirements

- Python 3.9 or higher
- GPU (optional, for advanced model switching)
- Internet connection for API calls

---

## Usage

1. Start the assistant:

   ```bash
   python main.py
   ```

2. Enter commands in the terminal or use voice input if enabled.

---

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set in the `.env` file.
2. **Model Loading Errors**: Install PyTorch or TensorFlow for `transformers` library.
3. **GPU Not Detected**: Verify GPU availability with:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Dependency Issues**: Reinstall dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

For further assistance, feel free to open an issue on the repository.
