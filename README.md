# REMI Personal Assistant

REMI is your multi-talented personal assistant designed to fetch weather updates, manage tasks, generate code, and so much more—all in one friendly package.

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

- **Weather Updates:** Get real-time weather info.
- **News Retrieval:** Stay informed with the latest news.
- **Translation:** Translate text seamlessly.
- **Image Search:** Find beautiful images on the fly.
- **Task Management:** Integrate with Todoist and Trello.
- **Messaging:** Send messages via Slack, Discord, and WhatsApp.
- **Directions:** Navigate with Google Maps.
- **Code Generation & Execution:** Write and run Python code effortlessly.
- **File and Database Management:** Create, move, or delete files and databases.
- **Personality Analysis:** Get insights about yourself.
- **Web Scraping:** Fetch store specials like a pro.
- **Advanced Model Switching:** Upgrade your AI model automatically when needed.

---

## Setup Instructions

### 1. Install Dependencies

Open your terminal and run:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
For GPU lovers, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Ensure Tesseract OCR is installed for PDF parsing.

### 2. Clone the Repository

Clone and navigate to the project directory:
```bash
git clone https://github.com/coff33ninja/remi.git
cd remi
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory and set your variables:
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
HUGGINGFACE_TOKEN=your_huggingface_token
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
COINGECKO_API_KEY=your_coingecko_api_key
TMDB_API_KEY=your_tmdb_api_key
NUTRITIONIX_API_KEY=your_nutritionix_api_key
NUTRITIONIX_APP_ID=your_nutritionix_app_id
SPOTIFY_API_KEY=your_spotify_api_key
SKYSCANNER_API_KEY=your_skyscanner_api_key
AMADEUS_API_KEY=your_amadeus_api_key
```

---

## Environment Variables and API Keys

For each service below, you must have the corresponding API key:

Hey DJ, no problem—let's integrate the Hugging Face token into the API keys table just like the others. Here's how the updated table might look:

---

### Environment Variables and API Keys

For each service below, you'll need the corresponding API key or token:

| Service            | API Key Variable             | Purpose                         | [Get API Key](#) |
|--------------------|------------------------------|---------------------------------|------------------|
| OpenWeatherMap     | `OPENWEATHERMAP_API_KEY`     | Weather updates                 | [OpenWeatherMap](https://home.openweathermap.org/users/sign_up) |
| NewsAPI            | `NEWSAPI_API_KEY`            | News articles                   | [NewsAPI](https://newsapi.org/register) |
| DeepL              | `DEEPL_API_KEY`              | Translation                     | [DeepL](https://www.deepl.com/pro#developer) |
| Unsplash           | `UNSPLASH_API_KEY`           | Image search                    | [Unsplash](https://unsplash.com/developers) |
| Todoist            | `TODOIST_API_KEY`            | Task management                 | [Todoist](https://developer.todoist.com/appconsole.html) |
| Trello             | `TRELLO_API_KEY`, `TRELLO_TOKEN` | Trello integration          | [Trello](https://trello.com/app-key) |
| Slack              | `SLACK_API_TOKEN`            | Slack messaging                 | [Slack](https://api.slack.com/apps) |
| Discord            | `DISCORD_BOT_TOKEN`          | Discord messaging               | [Discord](https://discord.com/developers/applications) |
| Wolfram Alpha      | `WOLFRAM_ALPHA_APP_ID`       | Query Wolfram Alpha             | [Wolfram Alpha](https://developer.wolframalpha.com/portal/myapps/) |
| Google Maps        | `GOOGLE_MAPS_API_KEY`        | Directions                      | [Google Maps](https://developers.google.com/maps/documentation/javascript/get-api-key) |
| Hugging Face       | `HUGGINGFACE_TOKEN`          | Model downloads and API calls   | [Hugging Face](https://huggingface.co/settings/tokens) |
| Alpha Vantage      | `ALPHA_VANTAGE_API_KEY`      | Stock prices                    | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) |
| CoinGecko          | `COINGECKO_API_KEY`          | Cryptocurrency rates            | [CoinGecko](https://www.coingecko.com/en/api) |
| TMDb               | `TMDB_API_KEY`               | Movie details                   | [TMDb](https://www.themoviedb.org/documentation/api) |
| Nutritionix        | `NUTRITIONIX_API_KEY`, `NUTRITIONIX_APP_ID` | Nutritional information | [Nutritionix](https://developer.nutritionix.com/) |
| Spotify            | `SPOTIFY_API_KEY`            | Music recommendations           | [Spotify](https://developer.spotify.com/documentation/web-api/) |
| Skyscanner         | `SKYSCANNER_API_KEY`         | Flight details                  | [Skyscanner](https://partners.skyscanner.net/affiliate-program) |
| Amadeus            | `AMADEUS_API_KEY`            | Hotel details                   | [Amadeus](https://developers.amadeus.com/register) |

---

## Supported Commands

### General Commands
- **Weather:** `What's the weather in [city]?`
- **News:** `Get news about [topic].`
- **Translate:** `Translate [text] to [language].`
- **Images:** `Search for images of [query].`
- **Tasks:** `Add task [task] due [date].`
- **Slack/Discord/WhatsApp:** `Send [message] to [platform].`
- **Directions:** `Get directions from [origin] to [destination].`

### Code Generation & Execution
- **Generate Code:** `Write Python code to [task].`
- **Execute Code:** `Run Python code [code].`
- **Explain Concept:** `Explain [concept] in Python.`

### File & Database Management
- **Create File:** `Create file [filename].`
- **Delete File:** `Delete file [filename].`
- **Move File:** `Move file [source] to [destination].`
- **Database:** `Create database [name].`

### Advanced Features
- **Model Switching:** Automatically upgrade if needed.
- **Personality Analysis:** `Tell me about myself.`
- **Web Scraping:** `Scrape specials from [store].`

---

## Data Placement

### Specials File
- **Text File Format:**
  `item:price:store:lat:lon`
  (Example: `milk:15.99:Shoprite:-33.9249:18.4241`)
- **PDF Files:**
  Contain text like `milk R15.99 at Shoprite lat -33.9249 lon 18.4241`.

### Database Files
- SQLite databases are auto-generated in the project directory.

### Model Training
1. Install prerequisites:
   ```bash
   pip install peft transformers torch
   ```
2. Run the training script:
   ```bash
   python train_lora.py
   ```
3. The fine-tuned model is saved in the `./fine_tuned_model` directory.

---

## System Requirements

- **Python:** 3.9 or higher
- **GPU:** Optional, for advanced model switching (recommended for performance)
- **Internet:** Required for API calls

---

## Usage

Start the assistant by running:
```bash
python main.py
```
Then, enter commands in the terminal or use voice input if enabled.

---

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   *Symptoms:* "API key not found" errors.
   *Fix:* Double-check your `.env` file against the [API Keys](#environment-variables-and-api-keys) section.

2. **Dependency Issues**
   *Symptoms:* Errors during installation or `ModuleNotFoundError`.
   *Fix:* Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **GPU Not Detected**
   *Symptoms:* "CUDA unavailable" warnings or slow CPU performance.
   *Fix:* Verify with:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   If it returns `False`, re-install PyTorch with:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Model Loading Errors**
   *Symptoms:* Errors with `transformers` or model initialization.
   *Fix:* Ensure your PyTorch or TensorFlow installation is CUDA-compatible.

---

### Detailed Troubleshooting: CUDA & `bitsandbytes`

If you run into errors about GPU support or issues with 4-bit quantization using `bitsandbytes`, follow these steps:

#### Problem Analysis
- **Scenario:** Your system (e.g., NVIDIA GeForce GTX 1060 with CUDA 12.8) shows that CUDA is installed, but `bitsandbytes` reports missing GPU support.
- **Issue:** Python libraries like PyTorch or `bitsandbytes` might not be configured to utilize CUDA properly.

#### Steps to Resolve

1. **Verify PyTorch CUDA Support**
   - Run:
     ```python
     import torch
     print(torch.__version__)
     print(torch.cuda.is_available())
     print(torch.version.cuda)
     ```
   - **Expected Output:**
     ```
     2.x.x+cu118  (or similar)
     True
     11.8         (or a version that matches your CUDA setup)
     ```
   - If `torch.cuda.is_available()` returns `False`, reinstall PyTorch with:
     ```bash
     pip uninstall torch
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     ```

2. **Fix `bitsandbytes` Installation**
   - *Issue:* "bitsandbytes was compiled without GPU support."
   - *Steps:*
     - Uninstall the current version:
       ```bash
       pip uninstall bitsandbytes
       ```
     - Install a CUDA-enabled version:
       ```bash
       pip install bitsandbytes --index-url https://pypi.org/simple/
       ```
     - If necessary, download a pre-built wheel from the [bitsandbytes GitHub releases](https://github.com/TimDettmers/bitsandbytes/releases) (e.g., `bitsandbytes-0.43.0-py3-none-win_amd64.whl` for Windows) and install:
       ```bash
       pip install path/to/bitsandbytes-0.43.0-py3-none-win_amd64.whl
       ```
     - Test it:
       ```python
       import bitsandbytes
       print(bitsandbytes.__version__)
       ```

3. **Check Environment Consistency**
   - Ensure your Python environment is set up correctly. On Windows, create and activate a virtual environment:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     ```

4. **Optional Fallback: Run on CPU**
   - If GPU support remains elusive, modify `train_lora.py` to run on CPU (at the cost of speed and increased memory use). Replace:
     ```python
     model = AutoModelForCausalLM.from_pretrained(
         model_name,
         quantization_config=BitsAndBytesConfig(load_in_4bit=True),
         device_map="auto",
         torch_dtype=torch.float16,
         token=os.getenv("HUGGINGFACE_TOKEN"),
     )
     ```
     With:
     ```python
     model = AutoModelForCausalLM.from_pretrained(
         model_name,
         device_map="cpu",
         torch_dtype=torch.float32,
         token=os.getenv("HUGGINGFACE_TOKEN"),
     )
     ```

5. **Run the Training Script Again**
   - After addressing the issues, execute:
     ```bash
     python train_lora.py
     ```
   - Make sure `dataset.txt` exists in the directory and follows the correct format.

#### Additional Notes
- **Hugging Face Token:** Ensure `HUGGINGFACE_TOKEN` is set in your `.env` file.
- **GPU Memory:** A GTX 1060 (6GB VRAM) should suffice for Mistral 7B with 4-bit quantization. Close unnecessary applications to free up memory.
- **Driver Update:** If problems persist, consider updating your NVIDIA driver from the [official site](https://www.nvidia.com/Download/index.aspx).
- **CUDA Compatibility:** Verify your CUDA version is compatible with the version of `bitsandbytes` in use. Consult [GitHub issues](https://github.com/TimDettmers/bitsandbytes/issues) for guidance.
