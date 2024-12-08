import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import sqlite3
import time
import os
from dotenv import load_dotenv
from termcolor import colored as col
import anthropic

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("Bot token not found in environment variables.")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Anthropic API key not found in environment variables.")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

DB_NAME = "messages.db"

def init_db():
    """Initializes the SQLite database and creates the messages table if it does not exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 chat_id INTEGER,
                 user_id INTEGER,
                 username TEXT,
                 message_text TEXT,
                 timestamp INTEGER
                 )''')
    
    columns = [col[1] for col in c.execute("PRAGMA table_info(messages)")]
    if 'username' not in columns:
        c.execute("ALTER TABLE messages ADD COLUMN username TEXT")
        conn.commit()
    conn.commit()
    conn.close()

def store_message(chat_id: int, user_id: int, username: str, message_text: str):
    """Stores a message in the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = int(time.time())
    c.execute("INSERT INTO messages (chat_id, user_id, username, message_text, timestamp) VALUES (?, ?, ?, ?, ?)",
              (chat_id, user_id, username, message_text, timestamp))
    conn.commit()

    c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
    count = c.fetchone()[0]
    if count > 1000:
        excess = count - 1000
        c.execute("""
            DELETE FROM messages 
            WHERE id IN (
                SELECT id FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
                LIMIT ?
            )
        """, (chat_id, excess))
        conn.commit()

    conn.close()

def get_last_n_messages(chat_id: int, n: int):
    """Fetches the last n messages from the chat."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT message_text, user_id, username FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?", (chat_id, n))
    rows = c.fetchall()
    conn.close()
    return [(r[0], r[1], r[2]) for r in reversed(rows)]

def call_claude_api(api_key, user_message, model="claude-3-5-sonnet-20240620", max_tokens=1024, system=""):
    """
    Function to call the Claude API and get a response.

    Parameters:
        api_key (str): The Claude API key.
        user_message (str): The message to send to Claude.
        model (str, optional): The Claude model to use.
        max_tokens (int, optional): The maximum number of tokens in the response.
        system (str, optional): System prompt or instructions.

    Returns:
        str: The model's response.
    """
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response_raw = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_message}],
            system=system
        )

        logger.info(col(str(response_raw.usage), "green"))

        response = response_raw.content[0].text

        if isinstance(response, str):
            return response
        else:
            raise ValueError(f"Got unrecognized response type {response}")
    except Exception as e:
        logger.error(f"Error communicating with Claude API: {e}")
        return "Sorry, there was an error processing your request."

def summarize_messages(messages):
    """
    Requests Claude to summarize the latest messages.
    Includes instructions to answer in the same language and use **bold** for important things.
    """
    if not messages:
        return "There are no messages to summarize."

    formatted_messages = "\n".join([f"{username if username else 'User '+str(user_id)}: {text}" for text, user_id, username in messages])

    prompt = (
        "Please provide a summary of the following chat conversation.\n\n"
        "Include what has been discussed recently, the most discussed topics, and who said what.\n"
        "Please answer using the same language as the conversation and use **bold** letters to highlight important things.\n\n"
        "Conversation:\n"
        f"{formatted_messages}\n\n"
        "Summary:"
    )

    summary = call_claude_api(
        api_key=ANTHROPIC_API_KEY,
        user_message=prompt,
        model=ANTHROPIC_MODEL,
        max_tokens=1024
    )
    return summary

def answer_question(messages, question):
    """
    Uses Claude to answer a question based on the conversation.
    Instruct Claude to use same language as the conversation and **bold** for important things.
    """
    if not messages:
        return "There are no messages to analyze."

    formatted_messages = "\n".join([f"{username if username else 'User '+str(user_id)}: {text}" for text, user_id, username in messages])

    prompt = (
        "Based on the following conversation, please answer the provided question.\n\n"
        "Please answer using the same language as the conversation and use **bold** letters to highlight important things.\n\n"
        f"Conversation:\n{formatted_messages}\n\n"
        f"Question: {question}\nAnswer:"
    )

    response = call_claude_api(
        api_key=ANTHROPIC_API_KEY,
        user_message=prompt,
        model=ANTHROPIC_MODEL,
        max_tokens=1024
    )
    return response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    await update.message.reply_text("Hello! I am a bot that can summarize and answer questions about the messages.", parse_mode='Markdown')

async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /summarize command."""
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) != 1:
        await update.message.reply_text("Usage: /summarize N", parse_mode='Markdown')
        return
    try:
        n = int(args[0])
    except ValueError:
        await update.message.reply_text("N must be a number.", parse_mode='Markdown')
        return

    if n <= 0 or n > 1000:
        await update.message.reply_text("N must be a number between 1 and 1000.", parse_mode='Markdown')
        return

    messages = get_last_n_messages(chat_id, n)
    logger.info(col(messages, 'green'))
    summary = summarize_messages(messages)
    await update.message.reply_text(summary, parse_mode='Markdown')

async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /ask command."""
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Usage: /ask your_question", parse_mode='Markdown')
        return

    question = " ".join(args)
    messages = get_last_n_messages(chat_id, 1000)
    response = answer_question(messages, question)
    await update.message.reply_text(response, parse_mode='Markdown')

async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Listens for non-command messages and stores them in the database."""
    message = update.effective_message
    chat_id = update.effective_chat.id
    user = message.from_user
    user_id = user.id if user else None
    username = user.username if user and user.username else (user.full_name if user else None)
    text = message.text

    if text is not None:
        store_message(chat_id, user_id, username, text)

def main():
    """Starts the bot."""
    init_db()

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("summarize", handle_summarize))
    application.add_handler(CommandHandler("ask", handle_ask))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_listener))

    application.run_polling()

if __name__ == '__main__':
    main()
