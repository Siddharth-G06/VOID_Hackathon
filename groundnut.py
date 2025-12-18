import os
import glob
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1️⃣ Set base dataset folder
# -------------------------------
# Change this path to your dataset location
base_dir = "Groundnut"

# Create train/test directories
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# -------------------------------
# 2️⃣ Split dataset into train/test
# -------------------------------
split_ratio = 0.8  # 80% train, 20% test

for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path):
        continue  # Skip non-folder files

    # Collect all image paths
    images = glob.glob(os.path.join(category_path, "*.jpg")) + \
             glob.glob(os.path.join(category_path, "*.png"))
    if not images:
        continue

    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create train/test subfolders
    train_cat_dir = os.path.join(train_dir, category)
    test_cat_dir = os.path.join(test_dir, category)
    os.makedirs(train_cat_dir, exist_ok=True)
    os.makedirs(test_cat_dir, exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(img, train_cat_dir)
    for img in test_images:
        shutil.copy(img, test_cat_dir)

print("✅ Dataset split into train/test sets successfully.")

# -------------------------------
# 3️⃣ Data generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------------
# 4️⃣ Model building (MobileNetV2)
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 5️⃣ Model training
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12
)

# -------------------------------
# 6️⃣ Save the model
# -------------------------------
model.save("groundnut_disease_model.h5")
print("\n✅ Model saved as 'groundnut_disease_model.h5'")
