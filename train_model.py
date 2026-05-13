import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ─── Fix 1: GPU memory issue ──────────────────────────────────────────────────
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ─── Fix 2: Find dataset path automatically ───────────────────────────────────
POSSIBLE_PATHS = [
    'archive/plantvillage dataset/color',
    'archive/plantvillage dataset/Color',
    'archive/PlantVillage dataset/color',
    'archive/PlantVillage/color',
    'archive/color',
    'plantvillage dataset/color',
    'dataset/color',
    'color',
]

DATASET_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        folders = [f for f in os.listdir(p)
                   if os.path.isdir(os.path.join(p, f))]
        if len(folders) >= 10:
            DATASET_PATH = p
            break

# ─── Fix 3: Search entire project if not found ────────────────────────────────
if DATASET_PATH is None:
    print("🔍 Searching for dataset folder...")
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d.lower() == 'color':
                full_path = os.path.join(root, d)
                try:
                    subfolders = [f for f in os.listdir(full_path)
                                  if os.path.isdir(os.path.join(full_path, f))]
                    if len(subfolders) >= 10:
                        DATASET_PATH = full_path
                        print(f"✅ Found dataset: {full_path}")
                        break
                except:
                    pass
        if DATASET_PATH:
            break

if DATASET_PATH is None:
    print("❌ Dataset not found automatically.")
    print("👉 Please set the path manually below:")
    DATASET_PATH = input("Enter full path to your color folder: ").strip()

print(f"\n✅ Dataset path : {DATASET_PATH}")

# ─── Verify dataset ───────────────────────────────────────────────────────────
all_folders = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])
print(f"✅ Total classes : {len(all_folders)}")
for i, f in enumerate(all_folders):
    count = len(os.listdir(os.path.join(DATASET_PATH, f)))
    print(f"   {i:02d}: {f}  ({count} images)")

# ─── Config ───────────────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16        # Fix 4: reduced from 32 to avoid memory error
EPOCHS      = 25

# ─── Data generators ──────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print("\n⏳ Loading training data...")
train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("⏳ Loading validation data...")
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ─── Save class names in EXACT index order ────────────────────────────────────
class_indices = train_gen.class_indices
class_names   = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)

NUM_CLASSES = len(class_names)
print(f"\n✅ Saved {NUM_CLASSES} classes to class_names.json")

# ─── Build model ──────────────────────────────────────────────────────────────
print("\n⏳ Building model...")

# Fix 5: clear any old model from memory first
tf.keras.backend.clear_session()

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x      = GlobalAveragePooling2D()(base_model.output)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model built")
print(f"   Input  : {model.input_shape}")
print(f"   Output : {model.output_shape}  ← must match {NUM_CLASSES}")

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'crop_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ─── Phase 1: Train top layers only ──────────────────────────────────────────
print("\n🚀 Phase 1: Training top layers (10 epochs)...")
print("   This will take 10-15 minutes...\n")
try:
    model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    print(f"❌ Phase 1 error: {e}")
    raise

# ─── Phase 2: Fine-tune last 30 layers ───────────────────────────────────────
print("\n🚀 Phase 2: Fine-tuning last 30 layers...")
print("   This will take 20-30 minutes...\n")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

try:
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
except Exception as e:
    print(f"❌ Phase 2 error: {e}")
    raise

# ─── Save & final report ──────────────────────────────────────────────────────
model.save('crop_disease_model.h5')
print("\n✅ Model saved: crop_disease_model.h5")

val_loss, val_acc = model.evaluate(val_gen, verbose=0)

print("\n" + "="*55)
print(f"  🎯 Validation Accuracy  : {val_acc  * 100:.2f}%")
print(f"  📉 Validation Loss      : {val_loss:.4f}")
print(f"  📦 Model output classes : {model.output_shape[-1]}")
print(f"  📋 class_names.json     : {NUM_CLASSES} classes")
print(f"  ✅ MATCH                : {model.output_shape[-1] == NUM_CLASSES}")
print("="*55)

if model.output_shape[-1] == NUM_CLASSES:
    print("\n🎉 Everything is correct! Now run: python app.py")
else:
    print("\n❌ Mismatch! Do not run app.py — retrain again.")