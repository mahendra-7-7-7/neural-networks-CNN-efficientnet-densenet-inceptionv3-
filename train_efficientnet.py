"""
EfficientNetB3 + CBAM Training Script (Chest X-ray Pneumonia Detection)
Optimized for NVIDIA RTX 3050 (6GB VRAM)
"""
import os, sys, cv2, shutil, random, time, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, mixed_precision
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# ── 1. GPU CONFIGURATION & MIXED PRECISION ─────────────────────────
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy('mixed_float16')
    print(f"[*] GPU Detected: {gpus[0].name} | TF Version: {tf.__version__} | Mixed Precision FP16 Enabled")
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
BATCH    = 8  # Guaranteed no OOM on 6GB VRAM

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
        improved = current > self.best if self.mode == 'max' else current < self.best
        if improved:
            self.best = current
            self.model.save_weights(self.filepath)
            if self.verbose:
                print(f"\n  [✓] {self.monitor}={current:.4f} -- Saved: {os.path.basename(self.filepath)}")

def apply_clahe(image):
    img  = image.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cl   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    # EfficientNetB3 has built-in Rescaling(1./255); pass [0, 255] float32
    return cv2.merge([cl, cl, cl]).astype(np.float32)

print("[*] Setting up Data Generators...")
train_gen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rotation_range=15, width_shift_range=0.12, height_shift_range=0.12,
    zoom_range=0.15, horizontal_flip=True, shear_range=0.1,
    brightness_range=[0.75, 1.25], fill_mode='nearest'
).flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', color_mode='rgb', shuffle=True, seed=42)

val_gen = ImageDataGenerator(preprocessing_function=apply_clahe
).flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', color_mode='rgb', shuffle=False)

STEPS     = train_gen.samples // BATCH
VAL_STEPS = val_gen.samples   // BATCH

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_gen.classes)
CLASS_WEIGHT = {0: cw[0], 1: cw[1]}

# ── CBAM + EFFICIENTNET-B3 ────────────────────────────────────────
L2 = 1e-4

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -tf.reduce_mean(alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss

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

def build_model():
    inp  = layers.Input(shape=(*IMG_SIZE, 3))
    base = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inp)
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
    return Model(inp, out, name='EfficientNetB3_CBAM'), base

print("[*] Building EfficientNetB3 + CBAM...")
model, base = build_model()

# ── STAGE 1: HEAD ONLY (6 epochs) ────────────────────────────────
print("\n" + "="*60)
print(" EFF STAGE 1: Head Only | Backbone Frozen | lr=1e-3 | Fast Warmup")
print("="*60)
base.trainable = False
model.compile(optimizer=Adam(1e-3), loss=focal_loss(),
              metrics=['accuracy', AUC(name='auc'),
                       Precision(name='precision'), Recall(name='recall')])

h1 = model.fit(
    train_gen, epochs=6, steps_per_epoch=STEPS,
    validation_data=val_gen, validation_steps=VAL_STEPS,
    class_weight=CLASS_WEIGHT,
    callbacks=[
        PureCheckpoint(os.path.join(OUTPUT_DIR,'models','stage1.h5'),
                       monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=3,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=2, min_lr=1e-10, verbose=1)
    ], verbose=1)

with open(os.path.join(OUTPUT_DIR,'models','history_s1.json'),'w') as f:
    json.dump({k:[float(v) for v in vals] for k,vals in h1.history.items()}, f, indent=2)

# ── STAGE 2: TOP 30% UNFROZEN (10 epochs) ────────────────────────
print("\n" + "="*60)
print(" EFF STAGE 2: Top 30% Backbone Unfrozen | lr=1e-4 | 10 epochs")
print("="*60)
base.trainable = True
cutoff = int(len(base.layers) * 0.70)
for i, l in enumerate(base.layers):
    if i < cutoff:
        l.trainable = False
    else:
        l.trainable = False if isinstance(l, tf.keras.layers.BatchNormalization) else True

model.compile(optimizer=Adam(1e-4), loss=focal_loss(),
              metrics=['accuracy', AUC(name='auc'),
                       Precision(name='precision'), Recall(name='recall')])

h2 = model.fit(
    train_gen, epochs=10, steps_per_epoch=STEPS,
    validation_data=val_gen, validation_steps=VAL_STEPS,
    callbacks=[
        PureCheckpoint(os.path.join(OUTPUT_DIR,'models','stage2.h5'),
                       monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=4,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=2, min_lr=1e-10, verbose=1)
    ], verbose=1)

with open(os.path.join(OUTPUT_DIR,'models','history_s2.json'),'w') as f:
    json.dump({k:[float(v) for v in vals] for k,vals in h2.history.items()}, f, indent=2)

# ── STAGE 3: ALL UNFROZEN (10 epochs) ────────────────────────────
print("\n" + "="*60)
print(" EFF STAGE 3: All Unfrozen (BN Frozen) | lr=5e-6 | 10 epochs")
print("="*60)
for l in model.layers:
    l.trainable = False if isinstance(l, tf.keras.layers.BatchNormalization) else True

model.compile(optimizer=Adam(5e-6), loss=focal_loss(),
              metrics=['accuracy', AUC(name='auc'),
                       Precision(name='precision'), Recall(name='recall')])

h3 = model.fit(
    train_gen, epochs=10, steps_per_epoch=STEPS,
    validation_data=val_gen, validation_steps=VAL_STEPS,
    callbacks=[
        PureCheckpoint(os.path.join(OUTPUT_DIR,'models','final_model.h5'),
                       monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=4,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                          patience=2, min_lr=1e-10, verbose=1)
    ], verbose=1)

with open(os.path.join(OUTPUT_DIR,'models','history_s3.json'),'w') as f:
    json.dump({k:[float(v) for v in vals] for k,vals in h3.history.items()}, f, indent=2)

print("\n" + "="*60)
print(f"[✓] EfficientNetB3 Training Completed!")
print(f"[*] Best val_auc: {max(h3.history['val_auc'])*100:.2f}% | val_acc: {max(h3.history['val_accuracy'])*100:.2f}%")
print(f"[*] Checkpoint saved to: C:\\pneu\\outputs\\models\\final_model.h5")
print("="*60)
