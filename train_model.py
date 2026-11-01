
import numpy as np
import tensorflow as tf
import os

from sklearn.utils.class_weight import compute_class_weight

# Load preprocessed images and labels
X = np.load("dataset/processed/X_images.npy")
y = np.load("dataset/processed/y_labels.npy")

# Load dataset split indices
splits = np.load("dataset/processed/split_indices.npz")

train_idx = splits["train_idx"]
val_idx = splits["val_idx"]
test_idx = splits["test_idx"]

# Prepare training, validation and testing data
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Balance the category weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

# Load MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build classification model
inputs = tf.keras.Input(shape=(160, 160, 3))

x = tf.keras.layers.Rescaling(
    scale=2.0,
    offset=-1.0
)(inputs)

x = tf.keras.layers.RandomFlip("horizontal")(x)
x = tf.keras.layers.RandomRotation(0.05)(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    6,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Create output folder
os.makedirs("models", exist_ok=True)

# Save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/water_pollution_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# Load best saved model
best_model = tf.keras.models.load_model(
    "models/water_pollution_model.keras"
)

# Evaluate on unseen test images
test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test
)

print("\nModel training completed!")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
