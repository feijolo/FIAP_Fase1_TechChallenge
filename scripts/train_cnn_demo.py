import os
from pathlib import Path
import json

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

DATA_DIR = Path('tech_challenge_phase1/data/cbis_ddsm_demo')
REPORT_DIR = Path('tech_challenge_phase1/reports')
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 6

if not DATA_DIR.exists():
    raise SystemExit(f'data dir not found: {DATA_DIR}')

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=10, width_shift_range=0.1,
                                   height_shift_range=0.1, horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

# build simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# save model and history
model.save(REPORT_DIR / 'image_demo_cnn.h5')
with open(REPORT_DIR / 'image_demo_cnn_history.json', 'w') as f:
    json.dump({k: [float(x) for x in v] for k,v in history.history.items()}, f)

# evaluate on validation set and save classification report
val_gen.reset()
Y_pred = model.predict(val_gen)
y_pred = (Y_pred.ravel() > 0.5).astype(int)
y_true = val_gen.classes

report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()), output_dict=True)
cm = confusion_matrix(y_true, y_pred)

pd.DataFrame(report).to_json(REPORT_DIR / 'image_demo_cnn_classification_report.json', orient='columns')
pd.DataFrame(cm).to_csv(REPORT_DIR / 'image_demo_cnn_confusion_matrix.csv', index=False)

print('Training complete. Reports and model saved to', REPORT_DIR)
