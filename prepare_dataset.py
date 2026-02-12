import os
import shutil
import pandas as pd

csv_file = "messidor_data.csv"
image_folder = "Data/train/image"

healthy_folder = "dataset/healthy"
abnormal_folder = "dataset/abnormal"

os.makedirs(healthy_folder, exist_ok=True)
os.makedirs(abnormal_folder, exist_ok=True)

data = pd.read_csv(csv_file)

for index, row in data.iterrows():
    image_name = row[0]
    grade = row[1]

    src = os.path.join(image_folder, image_name)

    if grade == 0:
        dst = os.path.join(healthy_folder, image_name)
    else:
        dst = os.path.join(abnormal_folder, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("Done! Images organized into healthy and abnormal.")
