from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import numpy as np
import numpy as np
import joblib
import cv2
import os

# Función para preprocesar la imagen (igual que antes)
def preprocesar_imagen(img):
    if img is None:
        return None
    
    # Convertir a espacio de color LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Ecualizar el histograma en el espacio LAB para mejorar el contraste
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)  # Ecualizar la componente L
    img_lab = cv2.merge([l, a, b])

    # Convertir de nuevo a BGR
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # Normalización de la imagen (opcional pero útil)
    img_normalizada = np.float32(img_bgr) / 255.0  # Escalar los valores de píxeles

    return img_normalizada

# Función para calcular el histograma de la imagen
def calcular_histograma(imagen):
    # Convertir a espacio de color LAB
    img_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)

    # Calcular el histograma para cada canal LAB
    hist_L = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
    hist_A = cv2.calcHist([img_lab], [1], None, [256], [-128, 128])
    hist_B = cv2.calcHist([img_lab], [2], None, [256], [-128, 128])

    # Normalizar los histogramas
    hist_L /= hist_L.sum()
    hist_A /= hist_A.sum()
    hist_B /= hist_B.sum()

    # Unificar los histogramas de los tres canales en un solo vector
    histograma_completo = np.concatenate([hist_L, hist_A, hist_B], axis=0)
    
    # Convertir a un vector unidimensional (de dimensión 1)
    return histograma_completo.flatten()

# Cargar el modelo KNN entrenado previamente (debe estar en el mismo archivo de código o en un archivo guardado)
def cargar_modelo_knn():
    # Aquí, deberías cargar el modelo KNN previamente entrenado o volver a entrenarlo si es necesario
    # Para simplicidad, asumimos que el modelo ya está entrenado y listo para usar
    # Si has guardado el modelo KNN previamente, puedes cargarlo usando pickle o joblib
    # Ejemplo: knn = joblib.load("modelo_knn.pkl")
    
    # Aquí simplemente volvemos a crear el KNN y lo entrenamos con los mismos datos de antes.
    # Este es un paso ilustrativo.
    return KNeighborsClassifier(n_neighbors=3)

# Función para obtener un color aleatorio
def get_random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())

# Función para asignar un color a cada clase de manera dinámica
def get_class_color(class_name, class_colors):
    if class_name not in class_colors:
        class_colors[class_name] = get_random_color()
    return class_colors[class_name]

def draw_detect(frame, x1, y1, x2, y2, bbox_color, label):
    # Calcular centroide del bounding box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centro del bbox

    # Dibujar un punto en el centroide en el color de la clase
    cv2.circle(frame, (cx, cy), radius=5, color=bbox_color, thickness=-1)
    #cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
    #cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

def data_detection(classification_model, roi, conf_detection):
    # Redimensionar la ROI a 640x640
    resized_roi = cv2.resize(roi, (128, 128))

    # Realizar clasificación con el modelo de clasificación
    classification_result = classification_model.predict(resized_roi, verbose=False, conf=conf_detection)
    
    # Obtener la clase de clasificación con mayor confianza
    class_id = classification_result[0].probs.top1
    class_name = classification_result[0].names[class_id]
    class_conf = float(classification_result[0].probs.top1conf)

    return class_name, class_conf

def data_detection_AVG_color(new_sample):
    # Cargar el modelo y el codificador de etiquetas
    model = tf.keras.models.load_model("rice_color_classifier.h5")
    label_encoder = joblib.load("label_encoder.pkl")

    # Obtener las probabilidades de predicción
    predictions = model.predict(new_sample)

    # Obtener la clase con mayor probabilidad
    predicted_class_index = np.argmax(predictions)

    # Decodificar el índice a la clase original
    predicted_class = label_encoder.inverse_transform([predicted_class_index])

    return predicted_class[0]


def generar_titulo(nombre_archivo):
    # Extraer la parte relevante del nombre del archivo (sin extensión)
    nombre_base = os.path.splitext(os.path.basename(nombre_archivo))[0]
    # Dividir el nombre en elementos separados por "_"
    name = nombre_base.split("_")
    # Eliminar los dos primeros elementos
    mezcla_relevante = name[2:]
    # Agrupar los elementos en pares
    mezclas = [mezcla_relevante[i:i+2] for i in range(0, len(mezcla_relevante), 2)]

    # Construir el título a partir de los pares encontrados
    titulo = ", ".join([f"{porcentaje}% {clase}" for [clase, porcentaje] in mezclas])
    print(titulo)
    return titulo
