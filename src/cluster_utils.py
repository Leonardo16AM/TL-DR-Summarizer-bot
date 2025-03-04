from datetime import datetime
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def cluster_to_text(messages):
    """
    Convierte un clúster de mensajes en un texto representativo
    para generar embeddings.
    """
    if not messages:
        return "" 
    # Construir texto del cluster sin ordenar por fecha
    cluster_text = []
    for msg in messages:
        sender = msg[2]
        text = msg[0]
        if text:
            cluster_text.append(f"{sender}: {text}")
    
    return "\n".join(cluster_text)

def is_active_conversation(messages, threshold_msgs_per_minute=0.1):
    """
    Determina si un clúster de mensajes representa una conversación activa
    basado en la frecuencia de mensajes por minuto.
    """
    if not messages or len(messages) < 20:  # Necesitamos al menos 20 mensajes para considerarla activa
        return False
    
    # Verificar si los mensajes tienen atributo date
    # En Neo4j puede venir como timestamp o date
    has_date = all(hasattr(msg, 'date') or hasattr(msg, 'timestamp') for msg in messages)
    
    if not has_date:
        logger.warning("Algunos mensajes no tienen atributo 'date', no se puede determinar si es activa")
        
        # Asumir que es activa si hay suficientes mensajes
        # Esto es una solución temporal - ajusta según sea necesario
        return len(messages) >= 20
    
    # Ordenar mensajes por fecha, adaptándose al formato disponible
    def get_message_time(msg):
        if hasattr(msg, 'date'):
            return msg.date
        elif hasattr(msg, 'timestamp'):
            # Si es un timestamp como string o número, convertirlo
            if isinstance(msg.timestamp, (int, float)):
                return datetime.fromtimestamp(msg.timestamp)
            elif isinstance(msg.timestamp, str):
                try:
                    return datetime.fromisoformat(msg.timestamp)
                except ValueError:
                    try:
                        return datetime.fromtimestamp(float(msg.timestamp))
                    except:
                        logger.error(f"No se pudo convertir timestamp: {msg.timestamp}")
                        return datetime.now()  # Fallback
        
        # Si llegamos aquí, no hay información de tiempo
        logger.warning(f"Mensaje sin información de tiempo: {msg}")
        return datetime.now()  # Fallback
    
    try:
        sorted_messages = sorted(messages, key=get_message_time)
        
        # Calcular tiempo transcurrido desde el primer al último mensaje
        first_time = get_message_time(sorted_messages[0])
        last_time = get_message_time(sorted_messages[-1])
        
        time_diff = last_time - first_time
        minutes_diff = time_diff.total_seconds() / 60
        
        # Evitar división por cero
        if minutes_diff < 0.1:
            minutes_diff = 0.1
        
        # Calcular tasa de mensajes por minuto
        msgs_per_minute = len(messages) / minutes_diff
        
        logger.info(f"Mensajes por minuto: {msgs_per_minute:.2f} ({len(messages)} msgs en {minutes_diff:.2f} mins)")
        
        return msgs_per_minute >= threshold_msgs_per_minute
    
    except Exception as e:
        logger.error(f"Error al calcular actividad de conversación: {e}")
        # En caso de error, asumimos que es activa si hay suficientes mensajes
        return len(messages) >= 10

def get_active_users_in_cluster(messages):
    """
    Retorna los usuarios que participaron activamente en el clúster.
    """
    if not messages:
        return []
    
    users = {}
    for msg in messages:
        # Usar getattr para manejar casos donde los atributos pueden no existir
        user_id = msg[1]
        username = msg[2]
        
        if user_id and user_id not in users:
            users[user_id] = username
    
    return [(user_id, username) for user_id, username in users.items() if user_id]