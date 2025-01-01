from nltk.corpus import stopwords
import re

from symspellpy import SymSpell, Verbosity
import os

class TextCleaner:
    def __init__(self, dictionary_path: str, max_edit_distance: int = 2):
        """
        Inicializa SymSpell con un diccionario predefinido.
        
        Parámetros:
        - dictionary_path (str): Ruta al archivo del diccionario (formato: palabra,frecuencia).
        - max_edit_distance (int): Distancia máxima de edición para sugerencias.
        """
        self.symspell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"El diccionario '{dictionary_path}' no existe.")
        
        # Cargar diccionario en SymSpell
        if not self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            raise RuntimeError("No se pudo cargar el diccionario.")
    

    def _correct_spelling(self, text: str) -> str:
        """
        Corrige errores ortográficos en un texto utilizando SymSpell.
        
        Parámetros:
        - text (str): Texto a corregir.
        
        Retorna:
        - str: Texto corregido.
        """
        corrected_words = []
        for word in text.split():
            suggestions = self.symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                # Usar la mejor sugerencia
                corrected_words.append(suggestions[0].term)
            else:
                # Si no hay sugerencias, dejar la palabra original
                corrected_words.append(word)
        return " ".join(corrected_words)
    
cleaner = TextCleaner(dictionary_path="spanish.txt")

def _clean_text(text: str, to_lowercase: bool = False, remove_special_chars: bool = False, 
                    correct_spelling: bool = False, remove_stopwords: bool = False, lang: str = "spanish") -> str:
    """
    Limpia el texto basado en los parámetros especificados.
    """
    cleaned_text = text

    # Convertir a minúsculas (opcional)
    if to_lowercase:
        cleaned_text = cleaned_text.lower()

    # Eliminar caracteres especiales (opcional)
    if remove_special_chars:
        # Nota: Mantener tildes y signos de puntuación básicos
        pattern = r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s.,;?!-]' if lang == "spanish" else r'[^a-zA-Z0-9\s.,;?!-]'
        cleaned_text = re.sub(pattern, '', cleaned_text)

    # Corrección ortográfica (opcional, usando SymSpell o similar)
    if correct_spelling:
        cleaned_text = cleaner._correct_spelling(cleaned_text)

    # Eliminar stopwords (opcional)
    if remove_stopwords:
        stop_words = set(stopwords.words(lang))
        print(stop_words)
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])

    # Normalizar espacios
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

S = ["Me duele la $ varriga AáaA",
"Las acciones relacionadas son las que integran el llamado sector agrícola. Todas las actividades económicas que abarca dicho sector tienen su fundamento en la explotación de los recursos que la tierra origina, favorecida por la acción del ser humano: alimentos vegetales como cereales, frutas, hortalizas, pastos cultivados y forrajes; fibras utilizadas por la industria textil; cultivos energéticos etc."]

for s in S:
    cs = _clean_text(s, to_lowercase = True, remove_special_chars = True, 
                    correct_spelling = True, remove_stopwords = True, lang = "english")
    print(cs)