# main.py

import logging
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
import os
import subprocess
from dotenv import load_dotenv
from termcolor import colored as col
import anthropic

import whisper 

# Importamos el DBManager de nuestro archivo aparte
from db_manager import DBManager

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Token de Telegram
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("Bot token not found in environment variables.")

# Claves de Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Anthropic API key not found in environment variables.")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

# Instanciamos la clase que maneja Neo4j
db_manager = DBManager()

def call_claude_api(api_key, user_message, model="claude-3-5-sonnet-20240620", 
                    max_tokens=1024, system=""):
    """
    Llamada al API de Claude para generar respuesta.
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
        logger.error(f"Error comunicando con la API de Claude: {e}")
        return "Sorry, there was an error processing your request."

def summarize_messages(messages):
    """
    Le pedimos a Claude que genere un resumen de la conversación.
    """
    if not messages:
        return "There are no messages to summarize."

    formatted_messages = "\n".join(
        [f"{username if username else 'User '+str(user_id)}: {text}"
         for text, user_id, username in messages]
    )

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
    Le pedimos a Claude que responda una pregunta basada en la conversación.
    """
    if not messages:
        return "There are no messages to analyze."

    formatted_messages = "\n".join(
        [f"{username if username else 'User '+str(user_id)}: {text}"
         for text, user_id, username in messages]
    )

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

def convert_to_wav(input_path, output_path):
    """
    Convierte un archivo de audio a formato WAV usando ffmpeg.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting to WAV: {e}")
        return False

def transcribe_with_local_whisper(file_path):
    """
    Carga el modelo local de Whisper y transcribe el archivo.
    """
    # Se puede cambiar el modelo a "base", "medium", "large", etc.
    model = whisper.load_model("small")
    
    # Transcribe localmente
    result = model.transcribe(file_path)
    # Extraemos el texto
    text = result["text"]
    return text.strip()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /start."""
    await update.message.reply_text(
        "Hello! I am a bot that can summarize and answer questions about the messages.\n"
        "Also, send me a voice note and I'll transcribe it using local Whisper!",
        parse_mode='Markdown'
    )

async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /summarize."""
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

    messages = db_manager.get_last_n_messages(chat_id, n)
    logger.info(col(messages, 'green'))
    summary = summarize_messages(messages)
    await update.message.reply_text(summary, parse_mode='Markdown')

async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /ask."""
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Usage: /ask your_question", parse_mode='Markdown')
        return

    question = " ".join(args)
    # Obtenemos los últimos 1000 mensajes
    messages = db_manager.get_last_n_messages(chat_id, 1000)
    response = answer_question(messages, question)
    await update.message.reply_text(response, parse_mode='Markdown')

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Maneja mensajes de voz (o audios) que llegan al bot:
      1. Descarga el archivo.
      2. Lo convierte a WAV.
      3. Llama a Whisper local para obtener la transcripción.
      4. Guarda la transcripción en la base de datos.
      5. Envía la transcripción como mensaje de respuesta.
    """
    message = update.effective_message
    chat_id = update.effective_chat.id
    telegram_message_id = message.message_id
    user = message.from_user
    user_id = user.id if user else None
    username = user.username if user and user.username else (user.full_name if user else None)

    # Dependiendo de si el audio llega como 'voice' o 'audio'
    if message.voice:
        file_id = message.voice.file_id
    elif message.audio:
        file_id = message.audio.file_id
    else:
        return  # No es ni voice ni audio

    # Descargamos el archivo OGG/MP3/lo que sea
    file = await context.bot.get_file(file_id)
    input_path = "temp_input_audio"
    output_path = "temp_output_audio.wav"

    await file.download_to_drive(custom_path=input_path)

    # Convertimos a WAV
    success = convert_to_wav(input_path, output_path)
    if not success:
        await update.message.reply_text("Error converting audio to WAV.")
        return

    # Llamamos a Whisper local para transcribir
    transcription = transcribe_with_local_whisper(output_path)

    # Guardamos la transcripción en la base de datos
    db_manager.add_message(
        chat_id=chat_id,
        telegram_message_id=telegram_message_id,
        user_id=user_id,
        username=username,
        message_text=transcription,
        reply_to_chat_id=None,
        reply_to_message_id=None
    )

    # Limpiamos archivos temporales
    try:
        os.remove(input_path)
        os.remove(output_path)
    except OSError:
        pass

    # Respondemos con el texto transcrito
    await update.message.reply_text(
        f"Transcription:\n{transcription}",
        parse_mode='Markdown'
    )

async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Escucha mensajes que no son comandos y los almacena en la base de datos.
    Verifica si es una respuesta (reply) a otro mensaje.
    """
    message = update.effective_message
    chat_id = update.effective_chat.id
    telegram_message_id = message.message_id
    user = message.from_user
    user_id = user.id if user else None
    username = user.username if user and user.username else (user.full_name if user else None)
    text = message.text

    if text is not None:
        # Si es un reply a otro mensaje de Telegram, 
        # obtener su message_id y chat_id para la relación
        reply_to_message = message.reply_to_message
        if reply_to_message:
            reply_to_chat_id = reply_to_message.chat.id
            reply_to_message_id = reply_to_message.message_id
        else:
            reply_to_chat_id = None
            reply_to_message_id = None

        # Guardamos el mensaje
        db_manager.add_message(
            chat_id=chat_id,
            telegram_message_id=telegram_message_id,
            user_id=user_id,
            username=username,
            message_text=text,
            reply_to_chat_id=reply_to_chat_id,
            reply_to_message_id=reply_to_message_id
        )

        # Controlamos que no exceda 1000
        count = db_manager.get_messages_count(chat_id)
        if count > 1000:
            excess = count - 1000
            db_manager.delete_oldest_messages(chat_id, excess)

def main():
    """Inicia el bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("summarize", handle_summarize))
    application.add_handler(CommandHandler("ask", handle_ask))

    # Handler para voice/audio
    application.add_handler(
        MessageHandler(filters.AUDIO | filters.VOICE, handle_voice_message)
    )

    # Handler para texto
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_listener)
    )

    application.run_polling()

    # Cerramos la conexión a Neo4j al final (opcional)
    db_manager.close()

if __name__ == '__main__':
    main()
