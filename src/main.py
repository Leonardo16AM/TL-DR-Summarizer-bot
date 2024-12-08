import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import sqlite3
import time
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("No se encontró el token del bot en las variables de entorno.")

DB_NAME = "messages.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 chat_id INTEGER,
                 user_id INTEGER,
                 message_text TEXT,
                 timestamp INTEGER
                 )''')
    conn.commit()
    conn.close()

def store_message(chat_id: int, user_id: int, message_text: str):
    """Almacena el mensaje en la BD e impone el límite de 1000 mensajes por grupo."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = int(time.time())
    c.execute("INSERT INTO messages (chat_id, user_id, message_text, timestamp) VALUES (?, ?, ?, ?)",
              (chat_id, user_id, message_text, timestamp))
    conn.commit()

    c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
    count = c.fetchone()[0]
    if count > 1000:
        exceso = count - 1000
        c.execute("DELETE FROM messages WHERE chat_id = ? ORDER BY id ASC LIMIT ?", (chat_id, exceso))
        conn.commit()

    conn.close()

def get_last_n_messages(chat_id: int, n: int):
    """Obtiene los últimos n mensajes del chat."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT message_text FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?", (chat_id, n))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in reversed(rows)]

def summarize_messages(messages):
    """Función placeholder para resumir una lista de mensajes."""
    # Por ahora, solo devolver un texto placeholder.
    return "Este es un resumen placeholder de los últimos mensajes."

def answer_question(messages, question):
    """Función placeholder para responder a una pregunta basada en mensajes."""
    # Por ahora, solo devolver un texto placeholder.
    return "Esta es una respuesta placeholder basada en los últimos mensajes."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola! Soy un bot para resumir y responder sobre mensajes.")

async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) != 1:
        await update.message.reply_text("Uso: /sumarize N")
        return
    try:
        n = int(args[0])
    except ValueError:
        await update.message.reply_text("N debe ser un número.")
        return

    if n <= 0 or n > 1000:
        await update.message.reply_text("N debe ser un número entre 1 y 1000.")
        return

    messages = get_last_n_messages(chat_id, n)
    resumen = summarize_messages(messages)
    await update.message.reply_text(resumen)

async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Uso: /ask pregunta")
        return

    question = " ".join(args)
    messages = get_last_n_messages(chat_id, 1000)
    respuesta = answer_question(messages, question)
    await update.message.reply_text(respuesta)

async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    chat_id = update.effective_chat.id
    user_id = message.from_user.id if message.from_user else None
    text = message.text

    if text is not None:
        store_message(chat_id, user_id, text)

def main():
    init_db()

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("summarize", handle_summarize))
    application.add_handler(CommandHandler("ask", handle_ask))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_listener))

    application.run_polling()

if __name__ == '__main__':
    main()
