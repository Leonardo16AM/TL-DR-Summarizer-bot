import sqlite3
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging
from termcolor import colored as col
import traceback

# Configuración de logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Modelo para generar embeddings - usar un modelo más pequeño y rápido
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo que ya está cargando según los logs

class InterestManager:
    def __init__(self, db_path="user_interests.db"):
        """
        Inicializa el manejador de intereses de usuarios.
        """
        logger.info(col("Inicializando InterestManager...", "cyan"))
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect_db()
        
        try:
            logger.info(col("Cargando modelo de embeddings...", "cyan"))
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(col("Modelo de embeddings cargado correctamente", "green"))
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info(col("InterestManager inicializado correctamente", "green"))
    
    def _connect_db(self):
        """
        Conecta a la base de datos SQLite.
        """
        try:
            logger.info(col(f"Conectando a la base de datos: {self.db_path}", "cyan"))
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self._init_db()
            logger.info(col("Conexión a base de datos establecida", "green"))
        except Exception as e:
            logger.error(f"Error al conectar a la base de datos: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _init_db(self):
        """
        Inicializa la base de datos con tablas necesarias.
        """
        try:
            logger.info(col("Inicializando esquema de base de datos...", "cyan"))
            # Tabla para usuarios
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                chat_id INTEGER,
                last_active TIMESTAMP
            )
            ''')
            
            # Tabla para embeddings de intereses
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interest_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                embedding BLOB,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
            
            self.conn.commit()
            logger.info(col("Esquema de base de datos inicializado correctamente", "green"))
        except Exception as e:
            logger.error(f"Error al inicializar la base de datos: {e}")
            logger.error(traceback.format_exc())
            raise
    def get_text_embedding(self, text):
        """
        Genera un embedding para un texto dado.
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Texto vacío, no se puede generar embedding")
            return None
            
        try:
            # Limitar longitud del texto para evitar problemas con SentenceTransformer
            max_length = 10000  # Ajustar según sea necesario
            if len(text) > max_length:
                logger.warning(f"Texto truncado de {len(text)} a {max_length} caracteres")
                text = text[:max_length]
                
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None
    
    def get_cluster_embedding(self, messages):
        """
        Genera un embedding para un cluster de mensajes.
        """
        if not messages:
            logger.warning("No hay mensajes en el cluster, no se puede generar embedding")
            return None
            
        try:
            # Concatenar todos los mensajes en un solo texto
            text = " ".join([getattr(msg, 'text', '') for msg in messages if hasattr(msg, 'text') and msg.text])
            if not text.strip():
                logger.warning("El cluster no contiene texto, no se puede generar embedding")
                return None
                
            logger.info(col(f"Generando embedding para cluster de {len(messages)} mensajes", "cyan"))
            return self.get_text_embedding(text)
        except Exception as e:
            logger.error(f"Error al procesar cluster para embedding: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def store_user_interest(self, user_id, username, chat_id, embedding):
        """
        Almacena el embedding de interés para un usuario.
        """
        if embedding is None:
            logger.warning(f"No se puede almacenar interés para usuario {username}: embedding nulo")
            return False
            
        try:
            logger.info(col(f"Almacenando interés para usuario {username} (ID: {user_id})", "cyan"))
            # Convertir embedding a bytes para guardar en SQLite
            embedding_bytes = embedding.tobytes()
            
            # Actualizar o insertar usuario
            self.cursor.execute(
                '''
                INSERT OR REPLACE INTO users (user_id, username, chat_id, last_active)
                VALUES (?, ?, ?, ?)
                ''', 
                (user_id, username, chat_id, datetime.now())
            )
            
            # Insertar embedding
            self.cursor.execute(
                '''
                INSERT INTO interest_embeddings (user_id, embedding, timestamp)
                VALUES (?, ?, ?)
                ''',
                (user_id, embedding_bytes, datetime.now())
            )
            
            # Limitar el número de embeddings por usuario (mantener solo los últimos 10)
            self.cursor.execute(
                '''
                DELETE FROM interest_embeddings 
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM interest_embeddings 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                )
                ''',
                (user_id, user_id)
            )
            
            self.conn.commit()
            logger.info(col(f"Interés almacenado correctamente para usuario {username}", "green"))
            return True
        except Exception as e:
            logger.error(f"Error al almacenar interés: {e}")
            logger.error(traceback.format_exc())
            if self.conn:
                self.conn.rollback()
            return False
    
    def get_user_embeddings(self, user_id):
        """
        Obtiene todos los embeddings de interés de un usuario.
        """
        try:
            logger.info(col(f"Obteniendo embeddings para usuario ID: {user_id}", "cyan"))
            self.cursor.execute(
                '''
                SELECT embedding FROM interest_embeddings
                WHERE user_id = ?
                ORDER BY timestamp DESC
                ''',
                (user_id,)
            )
            
            results = self.cursor.fetchall()
            embeddings = []
            
            for row in results:
                # Convertir bytes a numpy array
                embedding = np.frombuffer(row[0], dtype=np.float32)
                embeddings.append(embedding)
                
            logger.info(col(f"Obtenidos {len(embeddings)} embeddings para usuario ID: {user_id}", "green"))
            return embeddings
        except Exception as e:
            logger.error(f"Error al obtener embeddings de usuario: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def get_chat_users(self, chat_id):
        """
        Obtiene todos los usuarios activos de un chat.
        """
        try:
            logger.info(col(f"Obteniendo usuarios del chat ID: {chat_id}", "cyan"))
            self.cursor.execute(
                '''
                SELECT user_id, username FROM users
                WHERE chat_id = ?
                ''',
                (chat_id,)
            )
            
            users = self.cursor.fetchall()
            logger.info(col(f"Obtenidos {len(users)} usuarios del chat ID: {chat_id}", "green"))
            return users
        except Exception as e:
            logger.error(f"Error al obtener usuarios del chat: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def calculate_interest_score(self, user_id, current_embedding):
        """
        Calcula el score de interés entre el embedding actual y los del usuario.
        """
        if current_embedding is None:
            logger.warning(f"No se puede calcular score para usuario {user_id}: embedding actual nulo")
            return 0
            
        try:
            logger.info(col(f"Calculando score de interés para usuario ID: {user_id}", "cyan"))
            user_embeddings = self.get_user_embeddings(user_id)
            if not user_embeddings:
                logger.info(col(f"Usuario ID: {user_id} no tiene embeddings previos", "yellow"))
                return 0
                
            # Calcular similitud del coseno con cada embedding del usuario
            similarities = []
            for emb in user_embeddings:
                similarity = self._cosine_similarity(current_embedding, emb)
                similarities.append(similarity)
                
            # Retornar la mayor similitud encontrada
            max_similarity = max(similarities) if similarities else 0
            logger.info(col(f"Score máximo de interés para usuario ID: {user_id}: {max_similarity}", "green"))
            return max_similarity
        except Exception as e:
            logger.error(f"Error al calcular score de interés: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def _cosine_similarity(self, a, b):
        """
        Calcula la similitud del coseno entre dos vectores.
        """
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception as e:
            logger.error(f"Error al calcular similitud del coseno: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def find_interested_users(self, chat_id, current_embedding, threshold=0.7):
        """
        Encuentra usuarios que podrían estar interesados en la conversación actual.
        """
        if current_embedding is None:
            logger.warning("No se pueden encontrar usuarios interesados: embedding actual nulo")
            return []
            
        try:
            logger.info(col(f"Buscando usuarios interesados en chat ID: {chat_id} con umbral {threshold}", "cyan"))
            users = self.get_chat_users(chat_id)
            interested_users = []
            
            for user_id, username in users:
                score = self.calculate_interest_score(user_id, current_embedding)
                logger.info(f"Usuario {username} (ID: {user_id}): score de interés {score}")
                
                if score >= threshold:
                    interested_users.append((user_id, username, score))
                    
            # Ordenar por score de mayor a menor
            interested_users = sorted(interested_users, key=lambda x: x[2], reverse=True)
            logger.info(col(f"Encontrados {len(interested_users)} usuarios interesados", "green"))
            return interested_users
        except Exception as e:
            logger.error(f"Error al buscar usuarios interesados: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def close(self):
        """
        Cierra la conexión a la base de datos.
        """
        try:
            logger.info(col("Cerrando conexión a la base de datos", "cyan"))
            if self.conn:
                self.conn.close()
                self.conn = None
                self.cursor = None
            logger.info(col("Conexión a base de datos cerrada", "green"))
        except Exception as e:
            logger.error(f"Error al cerrar la base de datos: {e}")
            logger.error(traceback.format_exc())