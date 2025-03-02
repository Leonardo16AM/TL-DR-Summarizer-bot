# main.py

import logging
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
from datetime import datetime
import os
from dotenv import load_dotenv
from termcolor import colored as col
import anthropic
from voice_to_text import transcribe_with_local_whisper, convert_to_wav
# Importamos el DBManager de nuestro archivo aparte
from db_manager import DBManager
import re
from perplexity import calculate_perplexity
from interest_manager import InterestManager
from cluster_utils import cluster_to_text, is_active_conversation, get_active_users_in_cluster

MAX_MESSAGES_LIMIT = 300
load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from interest_manager import InterestManager
from cluster_utils import cluster_to_text, is_active_conversation, get_active_users_in_cluster
import traceback

# Inicializa el InterestManager después del DBManager
# Esta es una parte crítica - poner el logger antes para detectar problemas
logger.info(col("Inicializando InterestManager...", "magenta"))
try:
    interest_manager = InterestManager()
    logger.info(col("InterestManager inicializado correctamente", "green"))
except Exception as e:
    logger.error(f"Error al inicializar InterestManager: {e}")
    logger.error(traceback.format_exc())
    # Continuar sin el InterestManager en vez de bloquear todo el bot
    interest_manager = None

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

#region call_claude_api
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
        logger.error(f"Error communicating with Claude API: {e}")
        return "Sorry, there was an error processing your request."

#region summarize_messages
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
        "Please answer using the same language as the conversation, it must be a plain text in the same languaje as the original messages\n\n"
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

#region answer_question
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

#region /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /start."""
    await update.message.reply_text(
        "Hello! I am a bot that can summarize and answer questions about group messages.",
        parse_mode='Markdown'
    )


#region escape_markdown
def escape_markdown(text: str) -> str:
    """
    Escapa los caracteres que pueden romper la interpretación
    de Markdown en Telegram.
    """
    pattern = r'([\*\_\`\[\]\(\)])'
    return re.sub(pattern, r'\\\1', text)

#region /summarize
async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja el comando /summarize.
    
    Este comando recibe un número N como argumento y genera
    un resumen de los últimos N mensajes de la base de datos.
    """
    chat_id = update.effective_chat.id
    args = context.args

    if len(args) != 1:
        n=300
    else:
        try:
            n = int(args[0])
        except ValueError:
            await update.message.reply_text(
                "N must be a number.",
                parse_mode="Markdown"
            )
            return

    if n <= 0 or n > MAX_MESSAGES_LIMIT:
        await update.message.reply_text(
            f"N must be a number between 1 and {MAX_MESSAGES_LIMIT}.",
            parse_mode="Markdown"
        )
        return

    messages = db_manager.get_last_n_messages(chat_id, n)
    logger.info(col(messages, "green"))

    summary = summarize_messages(messages)
    summary = escape_markdown(summary)

    await update.message.reply_text(
        summary,
        parse_mode="Markdown"
    )

#region /ask
async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /ask."""
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Usage: /ask your_question", parse_mode='Markdown')
        return

    question = " ".join(args)
    # Obtenemos los últimos MAX_MESSAGES_LIMIT mensajes
    messages = db_manager.get_last_n_messages(chat_id, MAX_MESSAGES_LIMIT)
    response = answer_question(messages, question)
    await update.message.reply_text(response, parse_mode='Markdown')

def not_emoji(text):
    for i in text:
        if i>='a' and i<='b':
            return True
    return False

#region find_probable_reply
def find_probable_reply(update,chat_id,text):
    messages=db_manager.get_last_n_messages(chat_id, 5)
    print(f">> {text}")
    best=None
    bestp=1000
    for message in messages:
        print(message)
        p=calculate_perplexity("datificate/gpt2-small-spanish",'Mensaje: '+message[0]+' Respuesta: '+text)
        print(f"{message[0]}:{p}")
        if not_emoji(message[0]) and p<bestp:
            bestp=p
            best=message
    if best:
        print(col(f"El mensaje '{text}' probablemente  fue una respuesta a: '{best[0]}'",'blue'))
        return (chat_id,best[1])

last_notified={}
#region message_listener
async def message_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Escucha mensajes que no son comandos y los almacena en la base de datos.
    Verifica si es una respuesta (reply) a otro mensaje.
    Analiza clusters activos y notifica a usuarios potencialmente interesados.
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
            # reply_to_message = find_probable_reply(update, chat_id, text)
            reply_to_message = None
            if reply_to_message:
                reply_to_chat_id = reply_to_message[0]
                reply_to_message_id = reply_to_message[1]
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

        # Controlamos que no exceda MAX_MESSAGES_LIMIT
        count = db_manager.get_messages_count(chat_id)
        if count > MAX_MESSAGES_LIMIT:
            excess = count - MAX_MESSAGES_LIMIT
            db_manager.delete_oldest_messages(chat_id, excess)  
        
        # Obtenemos el último cluster de mensajes
        messages = db_manager.get_last_cluster(chat_id, 20, 30)
        logger.info(col(f"Obtenido cluster con {len(messages)} mensajes", "cyan"))
        
        # Verificamos si la conversación está activa
        if is_active_conversation(messages):
            logger.info(col("Conversación activa detectada", "green"))
            
            # Obtener el texto del cluster para generar embedding
            cluster_text = cluster_to_text(messages)
            logger.info(col(f"Texto del cluster generado ({len(cluster_text)} caracteres)", "cyan"))
            
            if not cluster_text:
                logger.warning("Cluster vacío, no se puede generar embedding")
                return
                
            current_embedding = interest_manager.get_text_embedding(cluster_text)
            
            if current_embedding is None:
                logger.warning("No se pudo generar embedding para el cluster")
                return
            
            # Guardar el interés para los usuarios activos en este cluster
            active_users = get_active_users_in_cluster(messages)
            logger.info(col(f"Usuarios activos en el cluster: {len(active_users)}", "cyan"))
            
            for active_user_id, active_username in active_users:
                if not active_user_id:
                    continue
                    
                success = interest_manager.store_user_interest(
                    active_user_id,
                    active_username,
                    chat_id,
                    current_embedding
                )
                if success:
                    logger.info(col(f"Guardado interés para {active_username or active_user_id}", "cyan"))
                else:
                    logger.warning(f"No se pudo guardar interés para usuario {active_username or active_user_id}")
            
            # Encontrar usuarios interesados
            threshold = 0.8  # Ajusta según sea necesario
            interested_users = interest_manager.find_interested_users(
                chat_id,
                current_embedding,
                threshold
            )
            print(interested_users)
            # Filtrar usuarios que ya están activos en el cluster actual
            active_user_ids = [user_id for user_id, _ in active_users]
            last_not = [user_id for user_id,_,_ in interested_users
                if last_notified.get(user_id) is not None 
                and abs(last_notified[user_id] - datetime.now().timestamp()) < 3600]

            print(last_not)

            users_to_notify = [
                (user_id, username) 
                for user_id, username, _ in interested_users 
                if user_id not in active_user_ids
                and user_id not in last_not
            ]
            print(users_to_notify)
            for user,_ in users_to_notify:
                last_notified[user]=datetime.now().timestamp()

            if users_to_notify:
                # Crear mensaje de notificación con @username (solo para los que tienen username)
                mentions = [f"@{username}" for _, username in users_to_notify if username]
                
                if mentions:
                    notification_text = "Esta conversación podría interesarles: " + " ".join(mentions)
                    
                    logger.info(col(f"Notificando a usuarios: {notification_text}", "yellow"))
                    
                    # Enviar notificación
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=notification_text,
                            reply_to_message_id=telegram_message_id
                        )
                    except Exception as e:
                        logger.error(f"Error al enviar notificación: {e}")
            else:
                logger.info("No hay usuarios para notificar")

#region handle_voice_message
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


#region main
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

    # Handler para texto - sólo agregar UNA vez
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_listener)
    )

    logger.info(col("Bot iniciado, iniciando polling...", "green"))
    application.run_polling()

    # Cerramos las conexiones a las bases de datos
    logger.info(col("Cerrando conexiones a bases de datos...", "yellow"))
    db_manager.close()
    if interest_manager is not None:
        interest_manager.close()

if __name__ == '__main__':
    main()
