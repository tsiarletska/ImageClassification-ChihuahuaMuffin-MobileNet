import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Enable eager execution for TensorFlow
tf.config.run_functions_eagerly(True)

train_path = os.path.join('data', 'train')
test_path = os.path.join('data', 'test')

# normalize the dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))  #normalization of images - train

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))  #normalization of images - test

total_train_samples = sum(1 for _ in train_data.unbatch())
train_size = int(total_train_samples * 0.8) 
val_size = total_train_samples - train_size  

unbatched_train_data = train_data.unbatch()
train = unbatched_train_data.take(train_size).batch(32)
val = unbatched_train_data.skip(train_size).take(val_size).batch(32)

print(f"Train size: {sum(1 for _ in train)} batches")
print(f"Validation size: {sum(1 for _ in val)} batches")
print(f"Test size: {sum(1 for _ in test_data)} batches")

# MobileNet
base_model = MobileNet(
    weights='imagenet',  
    include_top=False, 
    input_shape=(256, 256, 3)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train
hist = model.fit(
    train,
    epochs=11,
    validation_data=val,
    callbacks=[tensorboard_callback, early_stop]
)

# plot training and validation loss
plt.figure()
plt.plot(hist.history['loss'], label='Loss', color='silver')
plt.plot(hist.history['val_loss'], label='Validation Loss', color='tan')
plt.title('Loss')
plt.legend(loc='upper left')
plt.show()

# plot training and validation accuracy
plt.figure()
plt.plot(hist.history['accuracy'], label='Accuracy', color='silver')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy', color='tan')
plt.title('Accuracy')
plt.legend(loc='upper left')
plt.show()

#class report
def evaluate_model_and_generate_report(test_data, model):
    y_true = []
    y_pred = []

    for images, labels in test_data:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((predictions > 0.5).astype(int).flatten())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Chihuahua', 'Muffin']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model_and_generate_report(test_data, model)
