"""
DenseNet121 + CBAM Training Script (Chest X-ray Pneumonia Detection)
Optimized for NVIDIA RTX 3050 (6GB VRAM)
"""
import os, sys, cv2, shutil, random, time, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# ── 1. GPU CONFIGURATION ──────────────────────────────────────────
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[*] GPU Detected: {gpus[0].name} | TF Version: {tf.__version__}")
else:
    print("[!] WARNING: No GPU detected, running on CPU!")

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

BASE_DIR   = r'C:\pneu\chest_xray'
OUTPUT_DIR = r'C:\pneu\outputs'
TRAIN_DIR  = os.path.join(BASE_DIR, 'train')
VAL_DIR    = os.path.join(BASE_DIR, 'val')
TEST_DIR   = os.path.join(BASE_DIR, 'test')
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'),  exist_ok=True)

IMG_SIZE = (260, 260)
BATCH    = 8  # Rock-solid stability on 6GB VRAM

# ── 2. CALLBACK UTILITIES ─────────────────────────────────────────
class PureCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_auc', mode='max', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = -np.inf if mode == 'max' else np.inf
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            for k, v in logs.items():
                if self.monitor in k:
                    val = v; break
        current = float(val) if val is not None else -np.inf
        improved = (current > self.best) if self.mode == 'max' else (current < self.best)
        if improved:
            self.best = current
            self.model.save_weights(self.filepath)
            if self.verbose:
                print(f"\n  [✓] {self.monitor}={current:.4f} -- Saved: {os.path.basename(self.filepath)}")

def apply_clahe(image):
    img  = image.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cl   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    return cv2.merge([cl, cl, cl]).astype(np.float32) / 255.0

# ── 3. DATA GENERATORS ────────────────────────────────────────────
print("[*] Setting up Data Generators...")
train_gen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
    zoom_range=0.20, horizontal_flip=True, shear_range=0.12,
    brightness_range=[0.70, 1.30], fill_mode='nearest'
).flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', color_mode='rgb', shuffle=True, seed=42)

val_gen = ImageDataGenerator(preprocessing_function=apply_clahe
).flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', color_mode='rgb', shuffle=False)

STEPS     = train_gen.samples // BATCH
VAL_STEPS = val_gen.samples   // BATCH

def mixup_generator(gen, alpha=0.20):
    while True:
        x1, y1 = next(gen)
        x2, y2 = next(gen)
        lam = np.random.beta(alpha, alpha)
        yield lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2

train_gen_mix = mixup_generator(train_gen, alpha=0.20)
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_gen.classes)
CLASS_WEIGHT = {0: cw[0], 1: cw[1]}
print(f"[*] Steps/epoch: {STEPS} | Val steps: {VAL_STEPS} | Class Weights: {CLASS_WEIGHT}")

# ── 4. MODEL ARCHITECTURE (DenseNet121 + CBAM) ─────────────────────
L2 = 1e-4

def channel_attention(x, ratio=8):
    ch = x.shape[-1]
    d1 = layers.Dense(ch//ratio, activation='relu', kernel_regularizer=regularizers.l2(L2))
    d2 = layers.Dense(ch, kernel_regularizer=regularizers.l2(L2))
    avg = layers.GlobalAveragePooling2D()(x)
    mx  = layers.GlobalMaxPooling2D()(x)
    out = layers.Activation('sigmoid')(layers.Add()([d2(d1(avg)), d2(d1(mx))]))
    return layers.Multiply()([x, layers.Reshape((1, 1, ch))(out)])

def spatial_attention(x):
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    mx  = layers.Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
    out = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(
          layers.Concatenate(axis=-1)([avg, mx]))
    return layers.Multiply()([x, out])

def build_densenet():
    inp  = layers.Input(shape=(*IMG_SIZE, 3))
    base = DenseNet121(include_top=False, weights='imagenet', input_tensor=inp)
    base.trainable = False
    x = channel_attention(base.output)
    x = spatial_attention(x)
    x = layers.GlobalAveragePooling2D()(x)
    for units, drop in [(512, 0.50), (256, 0.40)]:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(L2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32',
                       kernel_regularizer=regularizers.l2(L2))(x)
    return Model(inp, out, name='DenseNet121_CBAM'), base

print("[*] Building DenseNet121 + CBAM...")
dn_model, dn_base = build_densenet()
print(f"[*] Total params: {dn_model.count_params():,}")

# ── 5. STAGE 1: HEAD ONLY TRAINING (15 epochs) ───────────────────
print("\n" + "="*60)
print(" STAGE 1: Head Only | Backbone Frozen | lr=1e-3")
print("="*60)
dn_base.trainable = False
dn_model.compile(optimizer=Adam(1e-3),
                 loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.10),
                 metrics=['accuracy', AUC(name='auc'),
                          Precision(name='precision'), Recall(name='recall')])

h1 = dn_model.fit(
    train_gen,
    epochs=15, steps_per_epoch=STEPS,
    validation_data=val_gen, validation_steps=VAL_STEPS,
    class_weight=CLASS_WEIGHT,
    callbacks=[
        PureCheckpoint(os.path.join(OUTPUT_DIR,'models','dn_stage1.h5'),
                       monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=8,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=4, min_lr=1e-10, verbose=1)
    ], verbose=1)

with open(os.path.join(OUTPUT_DIR,'models','dn_history_s1.json'),'w') as f:
    json.dump({k:[float(v) for v in vals] for k,vals in h1.history.items()}, f, indent=2)
print(f"[✓] Stage 1 Best val_auc: {max(h1.history['val_auc'])*100:.2f}% | val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

# ── 6. STAGE 2: TOP 50% UNFROZEN + MIXUP (25 epochs) ─────────────
print("\n" + "="*60)
print(" STAGE 2: Top 50% Backbone Unfrozen + Mixup | lr=1e-4")
print("="*60)
dn_base.trainable = True
cutoff = int(len(dn_base.layers) * 0.50)
for i, l in enumerate(dn_base.layers):
    if i < cutoff:
        l.trainable = False
    else:
        l.trainable = False if isinstance(l, tf.keras.layers.BatchNormalization) else True

unfrozen = sum(1 for l in dn_base.layers if l.trainable)
print(f"[*] Unfrozen layers: {unfrozen}/{len(dn_base.layers)} (BatchNormalization always frozen)")

dn_model.compile(optimizer=Adam(1e-4),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy', AUC(name='auc'),
                          Precision(name='precision'), Recall(name='recall')])

h2 = dn_model.fit(
    train_gen_mix,
    epochs=25, steps_per_epoch=STEPS,
    validation_data=val_gen, validation_steps=VAL_STEPS,
    callbacks=[
        PureCheckpoint(os.path.join(OUTPUT_DIR,'models','dn_stage2.h5'),
                       monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=8,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=4, min_lr=1e-10, verbose=1)
    ], verbose=1)

with open(os.path.join(OUTPUT_DIR,'models','dn_history_s2.json'),'w') as f:
    json.dump({k:[float(v) for v in vals] for k,vals in h2.history.items()}, f, indent=2)
print(f"[✓] Stage 2 Best val_auc: {max(h2.history['val_auc'])*100:.2f}% | val_acc: {max(h2.history['val_accuracy'])*100:.2f}%")

# ── 7. STAGE 3: FULL FINE-TUNE + COSINE DECAY (25 epochs) ────────
print("\n" + "="*60)
print(" STAGE 3: Full Fine-Tune (BN Frozen) | CosineDecay lr=1e-5")
print("="*60)
for l in dn_model.layers:
    l.trainable = False if isinstance(l, tf.keras.layers.BatchNormalization) else True

S3_EPOCHS = 25
cosine_lr = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1e-5, decay_steps=STEPS * S3_EPOCHS, alpha=1e-7)

dn_model.compile(optimizer=Adam(cosine_lr),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy', AUC(name='auc'),
                          Precision(name='precision'), Recall(name='recall')])

h3 = dn_model.fit(
    train_gen_mix,
    epochs=S3_EPOCHS, steps_per_epoch=STEPS,
    validation_data=val_gen, validation_steps=VAL_STEPS,
    callbacks=[
        PureCheckpoint(os.path.join(OUTPUT_DIR,'models','dn_stage3_best.h5'),
                       monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=10,
                      restore_best_weights=True, mode='max', verbose=1)
    ], verbose=1)

# Copy to dn_final.h5 for dual-compatibility
shutil.copy2(os.path.join(OUTPUT_DIR,'models','dn_stage3_best.h5'),
             os.path.join(OUTPUT_DIR,'models','dn_final.h5'))

with open(os.path.join(OUTPUT_DIR,'models','dn_history_s3.json'),'w') as f:
    json.dump({k:[float(v) for v in vals] for k,vals in h3.history.items()}, f, indent=2)

print("\n" + "="*60)
print(f"[✓] DenseNet121 Training Completed!")
print(f"[*] Final Stage 3 Best val_auc: {max(h3.history['val_auc'])*100:.2f}% | val_acc: {max(h3.history['val_accuracy'])*100:.2f}%")
print(f"[*] Checkpoints saved to: C:\\pneu\\outputs\\models\\dn_stage3_best.h5")
print("="*60)
