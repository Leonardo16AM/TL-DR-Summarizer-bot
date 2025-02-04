# TL-DR-Summarizer-bot
TLDR: Telegram bot to summarize group messages


This Telegram bot provides features for summarizing conversations, answering questions based on group messages, and transcribing voice messages. It integrates with Anthropic's Claude API for AI-driven responses and uses Neo4j for efficient database management.

## Features

- **Message Summarization**: Generates a summary of recent messages in a group.
- **Question Answering**: Responds to user queries based on previous group conversations.
- **Voice Transcription**: Transcribes voice messages using a locally deployed Whisper model.

## Installation

Follow these steps to set up the bot on your local machine.

### Prerequisites

Ensure you have the following installed:

1. **Python 3.9+**
2. **Neo4j**: Download and install from [Neo4j Download Center](https://neo4j.com/download/).
3. **FFmpeg**: Download and install from [FFmpeg Official Site](https://ffmpeg.org/download.html).

### Environment Variables

Create a `.env` file in the `src` directory with the following variables:

```env
# Telegram Bot Token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL="claude-3-haiku-20240307"

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### Clone the Repository

```bash
git clone https://github.com/your-username/telegram-bot.git
cd telegram-bot
```

### Install Dependencies

Install required Python packages using `pip` or `pipenv`:

```bash
pip install -r requirements.txt
```
### Start Neo4j Database

1. Launch the Neo4j database server.
2. Configure the `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in the `.env` file to match your Neo4j instance.

### Run the Bot

```bash
cd src
python main.py
```

## Usage

### Commands

- **`/start`**: Start a conversation with the bot.
- **`/summarize [N]`**: Summarize the last `N` messages in the group (default: 300).
- **`/ask [question]`**: Ask a question based on the group's conversation.

### Voice Message Handling

The bot automatically processes and transcribes voice messages, saving the transcription to the database and replying with the text.

## Notes

- Ensure the Neo4j server is running before starting the bot.
- FFmpeg must be accessible in your system's PATH.
- Update the `.env` file with appropriate credentials before running the bot.

## Contributing

Feel free to open issues and submit pull requests to improve the bot.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
