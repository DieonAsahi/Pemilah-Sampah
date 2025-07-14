import os
import shutil
import json
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ====== 1. Salin Semua Gambar ke Folder Gabungan ======
folder_map_utama = {
    'Recyclable': 'daur_ulang',
    'Non-Recyclable': 'anorganik',
    'Organic': 'organik',
    'Hazardous': 'b3'
}

def copy_all_images_recursive(src_root, label, dst_root):
    dst = os.path.join(dst_root, label)
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst, file)
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    new_name = f"{name}_{uuid.uuid4().hex[:6]}{ext}"
                    dst_file = os.path.join(dst, new_name)
                shutil.copy2(src_file, dst_file)

base_dir = 'combined_dataset'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

for folder, label in folder_map_utama.items():
    src = os.path.join('dataset', folder)
    if os.path.exists(src):
        copy_all_images_recursive(src, label, base_dir)

# ====== 2. Cek distribusi data ======
print("\n📊 Distribusi Dataset:")
total_images = 0
for cls in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls)
    if os.path.isdir(cls_path):
        count = sum(1 for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        total_images += count
        print(f"- {cls}: {count} gambar")
print(f"Total gambar: {total_images}\n")

# ====== 3. Image Preprocessing ======
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ====== 4. Hitung Class Weight ======
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ====== 5. Model Transfer Learning (MobileNetV2) ======
IMG_SHAPE = IMG_SIZE + (3,)
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False  # Bekukan pretrained layer untuk training awal

# Tambahkan layer klasifikasi di atas base_model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Kompilasi dengan learning rate kecil
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ====== 6. Training ======
os.makedirs("model", exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("model/best_model.h5", save_best_only=True, monitor="val_accuracy")
]

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# ====== 7. Evaluasi ======
print("\n📈 Evaluasi Confusion Matrix & Classification Report:")
val_gen.reset()
y_pred = model.predict(val_gen, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
true_labels = val_gen.classes

class_names = list(train_gen.class_indices.keys())
conf_matrix = confusion_matrix(true_labels, y_pred_labels)
report = classification_report(true_labels, y_pred_labels, target_names=class_names, zero_division=0)
accuracy = accuracy_score(true_labels, y_pred_labels)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
print(f"\n🎯 Akurasi Validasi Akhir: {accuracy:.4f}")

# Simpan evaluasi ke file
with open("model/evaluation.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\n\nAkurasi: {accuracy:.4f}\n")

# ====== 8. Simpan Model ======
model.save("model/model.h5")
# model.save("model/model.keras")  # format baru
with open("model/label_map.json", "w") as f:
    json.dump(train_gen.class_indices, f)
with open("model/model_architecture.json", "w") as f:
    f.write(model.to_json())
with open("model/training_history.json", "w") as f:
    json.dump(history.history, f)

print("\n✅ Model berhasil dilatih dan disimpan.")
print(f"📂 Label Map: {train_gen.class_indices}")

# ====== 9. Fine-Tuning Lanjutan (Opsional) ======
print("\n🔧 Fine-tuning lanjutan MobileNetV2 dimulai...")

# Buka semua layer MobileNetV2 agar dapat dilatih ulang
base_model.trainable = True

# Bekukan sebagian besar layer awal (yang sangat general)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Kompilasi ulang model dengan learning rate lebih kecil
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tambah training untuk fine-tuning
fine_tune_history = model.fit(
    train_gen,
    epochs=5,  # Tambahan epoch setelah tahap awal
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# Evaluasi kembali setelah fine-tuning
print("\n📈 Evaluasi Setelah Fine-Tuning:")
val_gen.reset()
y_pred_ft = model.predict(val_gen, verbose=0)
y_pred_labels_ft = np.argmax(y_pred_ft, axis=1)
true_labels_ft = val_gen.classes

conf_matrix_ft = confusion_matrix(true_labels_ft, y_pred_labels_ft)
report_ft = classification_report(true_labels_ft, y_pred_labels_ft, target_names=class_names, zero_division=0)
accuracy_ft = accuracy_score(true_labels_ft, y_pred_labels_ft)

print("Confusion Matrix:")
print(conf_matrix_ft)
print("\nClassification Report:")
print(report_ft)
print(f"\n🎯 Akurasi Validasi Akhir Setelah Fine-Tuning: {accuracy_ft:.4f}")

# Simpan ulang evaluasi
with open("model/evaluation_finetune.txt", "w") as f:
    f.write("Confusion Matrix (Fine-Tuning):\n")
    f.write(str(conf_matrix_ft))
    f.write("\n\nClassification Report (Fine-Tuning):\n")
    f.write(report_ft)
    f.write(f"\n\nAkurasi Setelah Fine-Tuning: {accuracy_ft:.4f}\n")

# Simpan ulang model hasil fine-tuning
model.save("model/final_model.h5")
print("\n✅ Model fine-tuning berhasil disimpan ke model/final_model.h5")

