#WAIUP - Position tag evaluation AI
# Author: Guilherme da Silva

#Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import xml.etree.ElementTree as ET

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

db_path = pathlib.Path(r'C:\Users\94gui\Documents\TesteJulio\data')

image_count = len(list(db_path.glob('**\*.jpeg')))
print(image_count)

tree = ET.parse(r'C:\Users\94gui\Documents\TesteJulio\annotations.xml')

images = list(tree.iter("image"))

tag_list = []

for i in images:
    image_name = i.attrib['name']
    image_tag = ""
    for tag in i.iter("tag"):
        if 'P_' in tag.attrib['label']:
            image_tag = tag.attrib['label']
            if image_tag == "P_NG":
                tag_list.append(0)
            elif image_tag == "P_OK":
                tag_list.append(1)
            break


tf.keras.preprocessing.image_dataset_from_directory(
    db_path,
    labels=tag_list,
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)
