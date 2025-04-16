import os
import pandas as pd
from tqdm import tqdm

# Ruta de entrada y salida
input_dir = 'ruta/a/tu/carpeta'  # Reemplazar con la ruta donde están las imágenes y _annotations.csv
output_dir = os.path.join(input_dir, 'yolo_labels')
os.makedirs(output_dir, exist_ok=True)

# Leer CSV
csv_path = os.path.join(input_dir, '_annotations.csv')
df = pd.read_csv(csv_path)

# Obtener lista de clases y asignarles un ID
classes = sorted(df['class'].unique())
class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

# Guardar archivo de clases
with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")

# Agrupar anotaciones por imagen
grouped = df.groupby('filename')

# Procesar cada imagen
for filename, group in tqdm(grouped):
    image_width = group.iloc[0]['width']
    image_height = group.iloc[0]['height']
    
    yolo_lines = []
    for _, row in group.iterrows():
        class_id = class_to_id[row['class']]
        
        # Coordenadas normalizadas
        x_center = ((row['xmin'] + row['xmax']) / 2) / image_width
        y_center = ((row['ymin'] + row['ymax']) / 2) / image_height
        width = (row['xmax'] - row['xmin']) / image_width
        height = (row['ymax'] - row['ymin']) / image_height
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Guardar archivo .txt
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))
