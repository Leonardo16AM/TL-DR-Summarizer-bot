import whisper 
import subprocess
import os
import logging


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)



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