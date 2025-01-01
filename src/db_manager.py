# db_manager.py

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"# db_manager.py
# db_manager.py

import os
import time
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from nltk.corpus import stopwords
import re
from symspellpy import SymSpell, Verbosity

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
    

load_dotenv()

class DBManager:
    def __init__(self):
        """
        Inicializa la conexión a la base de datos Neo4j y 
        el modelo local de embeddings (Sentence BERT).
        """
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        # Inicializar driver de Neo4j
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Cargar modelo local de embeddings (puedes cambiar por otro)
        # Ejemplo: 'bert-base-nli-mean-tokens', 'all-MiniLM-L6-v2', etc.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Inicializar limpiador de texto
        self.cleaner = TextCleaner(dictionary_path="spanish+english.txt")

        # Opcional: Crear índices o constraints en Neo4j al iniciar
        self._create_schema()

    def _clean_text(self, text: str, to_lowercase: bool = False, remove_special_chars: bool = False, 
                    correct_spelling: bool = False, remove_stopwords: bool = False, lang: str = "spanish+english") -> str:
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
            pattern = r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s.,;?!-]' if lang in ["spanish", "spanish+english"] else r'[^a-zA-Z0-9\s.,;?!-]'
            cleaned_text = re.sub(pattern, '', cleaned_text)

        # Corrección ortográfica (opcional, usando SymSpell o similar)
        if correct_spelling:
            cleaned_text = self.cleaner._correct_spelling(cleaned_text)

        # Eliminar stopwords (opcional)
        if remove_stopwords:
            stop_words = []
            if lang == "spanish+english":
                stop_words_spanish = set(stopwords.words("spanish"))
                stop_words_english = set(stopwords.words("english"))
                stop_words = stop_words_spanish.union(stop_words_english)
            else:
                stop_words = set(stopwords.words(lang))
            cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])

        # Normalizar espacios
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    def close(self):
        """ Cierra la conexión al driver de Neo4j. """
        self.driver.close()

    def _create_schema(self):
        """
        Crea índices o constraints necesarios en Neo4j 
        (opcional, pero recomendado para performance).
        """
        with self.driver.session() as session:
            # Crear un índice único para cada mensaje basado en 'id'
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE")
            # Crear un índice para 'chat_id' para optimizar búsquedas
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.chat_id)")

    def add_message(self, chat_id: int, telegram_message_id: int, user_id: int, username: str, message_text: str, 
                   reply_to_chat_id: int = None, reply_to_message_id: int = None):
        """
        Añade un mensaje a la base de datos, junto con su embedding.
        Si 'reply_to_message_id' está presente, crea la relación (m)-[:REPLIES_TO]->(replyMsg).
        """
        timestamp = int(time.time())
        embedding = self._compute_embedding(message_text)
        
        # Limpiar mensaje
        clean_message_text = self._clean_text(message_text, to_lowercase = True, remove_special_chars = True, correct_spelling = True, remove_stopwords = False, lang = "spanish+english")

        # Crear un identificador único para el mensaje usando chat_id y telegram_message_id
        message_id = f"{chat_id}_{telegram_message_id}"

        # Si el mensaje es una respuesta, crear el identificador único del mensaje al que responde
        if reply_to_chat_id and reply_to_message_id:
            reply_to_db_id = f"{reply_to_chat_id}_{reply_to_message_id}"
        else:
            reply_to_db_id = None

        with self.driver.session() as session:
            # Crear nodo del mensaje
            session.run(
                """
                CREATE (m:Message {
                    id: $message_id,
                    chat_id: $chat_id,
                    telegram_message_id: $telegram_message_id,
                    user_id: $user_id,
                    username: $username,
                    text: $message_text,
                    clean_text: $clean_message_text,
                    timestamp: $timestamp,
                    embedding: $embedding
                })
                """,
                message_id=message_id,
                chat_id=chat_id,
                telegram_message_id=telegram_message_id,
                user_id=user_id,
                username=username,
                message_text=message_text,
                clean_message_text=clean_message_text,
                timestamp=timestamp,
                embedding=embedding
            )

            # Si existe un reply_to_message_id, crear la relación
            if reply_to_db_id:
                session.run(
                    """
                    MATCH (m:Message {id: $message_id})
                    MATCH (r:Message {id: $reply_to_db_id})
                    MERGE (m)-[:REPLIES_TO]->(r)
                    """,
                    message_id=message_id,
                    reply_to_db_id=reply_to_db_id
                )

    def get_last_n_messages(self, chat_id: int, n: int):
        """
        Retorna los últimos n mensajes de un chat, ordenados por timestamp DESC.
        Devuelve la lista en orden cronológico ASC (del más viejo al más nuevo).
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Message {chat_id: $chat_id})
                RETURN m
                ORDER BY m.timestamp DESC
                LIMIT $limit
                """,
                chat_id=chat_id,
                limit=n
            )

            # Revertir el orden para tener cronología ascendente
            messages = []
            for record in result:
                m = record["m"]
                messages.append( (m["text"], m["user_id"], m["username"]) )
            
            messages.reverse()
            return messages

    def get_messages_count(self, chat_id: int):
        """ Retorna el número de mensajes en un chat dado. """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Message {chat_id: $chat_id})
                RETURN count(m) as count
                """,
                chat_id=chat_id
            )
            return result.single()["count"]

    def delete_oldest_messages(self, chat_id: int, excess: int):
        """
        Elimina los 'excess' mensajes más antiguos en un chat dado.
        """
        with self.driver.session() as session:
            # Eliminar mensajes más antiguos basados en timestamp ASC
            session.run(
                """
                MATCH (m:Message {chat_id: $chat_id})
                WITH m
                ORDER BY m.timestamp ASC
                LIMIT $excess
                DETACH DELETE m
                """,
                chat_id=chat_id,
                excess=excess
            )

    def get_similar_messages(self, message_text: str, top_k: int = 5):
        """
        Retorna los mensajes más similares (por coseno de embeddings).
        Calcula el embedding del 'message_text', compara con todos, 
        ordena por similitud y devuelve los top_k.
        """
        # Calculamos embedding local
        query_emb = self._compute_embedding(message_text)

        # Descargamos todos los mensajes con su embedding
        messages_list = []
        with self.driver.session() as session:
            result = session.run("MATCH (m:Message) RETURN m")
            for record in result:
                m = record["m"]
                if m["embedding"] is not None:
                    messages_list.append({
                        "id": m["id"],
                        "text": m["text"],
                        "user_id": m["user_id"],
                        "username": m["username"],
                        "embedding": m["embedding"]
                    })

        # Calculamos similitud en Python
        scored = []
        for msg in messages_list:
            emb = np.array(msg["embedding"], dtype=np.float32)
            score = self._cosine_similarity(query_emb, emb)
            scored.append((msg, score))

        # Ordenamos por score descendente
        scored.sort(key=lambda x: x[1], reverse=True)

        # Tomamos los top_k
        top_messages = scored[:top_k]

        # Retornamos en un formato conveniente (mensaje + similitud)
        return [(t[0], t[1]) for t in top_messages]

    def _compute_embedding(self, text: str):
        """
        Devuelve el embedding de un texto como lista float con Sentence BERT.
        """
        emb = self.embedding_model.encode([text], convert_to_numpy=True)[0]
        return emb.tolist()

    def _cosine_similarity(self, v1, v2):
        """
        Retorna la similitud coseno de v1 y v2.
        """
        v1 = np.array(v1, dtype=np.float32)
        v2 = np.array(v2, dtype=np.float32)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dot / (norm1 * norm2 + 1e-10)
