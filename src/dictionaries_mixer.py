import math

def cargar_diccionario(archivo):
    """Carga un diccionario de frecuencias desde un archivo."""
    diccionario = {}
    with open(archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            try:
                palabra, frecuencia = linea.strip().split()
                diccionario[palabra] = int(frecuencia)
            except ValueError:
              print(f"Línea ignorada por formato incorrecto: {linea.strip()}")
    return diccionario

def combinar_diccionarios(diccionario_es, diccionario_en):
    """Combina dos diccionarios de frecuencias y calcula el promedio truncado."""
    combinado = {}
    palabras = set(diccionario_es.keys()) | set(diccionario_en.keys())  # Unión de conjuntos para evitar duplicados

    for palabra in palabras:
      frecuencia_es = diccionario_es.get(palabra, 0) #si no existe, da 0
      frecuencia_en = diccionario_en.get(palabra, 0) #si no existe, da 0
      
      suma = frecuencia_es + frecuencia_en

      combinado[palabra] = suma
    return combinado

def guardar_diccionario(diccionario, archivo):
    """Guarda un diccionario de frecuencias en un archivo."""
    with open(archivo, 'w', encoding='utf-8') as f:
        for palabra, frecuencia in diccionario.items():
            f.write(f"{palabra} {frecuencia}\n")

if __name__ == "__main__":
    archivo_es = "spanish.txt"
    archivo_en = "english.txt"
    archivo_salida = "spanish+english.txt"

    # Cargar diccionarios
    diccionario_es = cargar_diccionario(archivo_es)
    diccionario_en = cargar_diccionario(archivo_en)

    # Combinar y calcular promedios
    diccionario_combinado = combinar_diccionarios(diccionario_es, diccionario_en)

    # Ordenar por frecuencia (descendente)
    diccionario_ordenado = dict(sorted(diccionario_combinado.items(), key=lambda item: item[1], reverse=True))

    # Guardar el diccionario combinado
    guardar_diccionario(diccionario_ordenado, archivo_salida)
    
    print(f"Diccionarios combinados y guardados en '{archivo_salida}'")
