import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load preprocessed data
X_train = np.load("data/preprocessed/X_train.npy")
X_val = np.load("data/preprocessed/X_val.npy")
X_test = np.load("data/preprocessed/X_test.npy")
y_train = np.load("data/preprocessed/y_train.npy")
y_val = np.load("data/preprocessed/y_val.npy")
y_test = np.load("data/preprocessed/y_test.npy")

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build model with transfer learning (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("model_best.keras", save_best_only=True)
]

# Train model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Save final model
model.save("model_final.keras")

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")