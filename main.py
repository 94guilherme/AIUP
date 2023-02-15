#WAIUP - Position tag evaluation AI
# Author: Guilherme da Silva

#Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

db_path = pathlib.Path(r'C:\Users\94gui\Documents\TesteJulio\data')

image_count = len(list(db_path.glob('**\*.jpeg')))
print(image_count)

