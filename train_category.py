import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===== CHANGE ONLY THIS LINE =====
# Options: 'Places', 'Basics', 'Days_and_Time', 'Jobs', 'Animals'
CATEGORY = 'Greetings'
# =================================

DATA_PATH = 'MP_Data'
MIN_CLIPS = 12  # words with fewer clips are ignored

if CATEGORY == 'Greetings':
    actions = ['Greetings_48. Hello', 'Greetings_53. Good evening']
else:
    actions = sorted(
        d for d in os.listdir(DATA_PATH)
        if d.startswith(CATEGORY + '_')
        and len(os.listdir(os.path.join(DATA_PATH, d))) >= MIN_CLIPS
    )

print(f"[INFO] Training '{CATEGORY}' with {len(actions)} words:")
for a in actions:
    print("   ", a)

label_map = {a: i for i, a in enumerate(actions)}
sequences, labels = [], []
skipped = 0

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for seq in os.listdir(action_path):
        try:
            window = [np.load(os.path.join(action_path, seq, f"{f}.npy")) for f in range(30)]
            sequences.append(window)
            labels.append(label_map[action])
        except Exception:
            skipped += 1

print(f"[INFO] Loaded clips: {len(sequences)}  |  Skipped (broken): {skipped}")

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=labels
)

# ---- Data augmentation (training clips only) ----
def augment(clip):
    c = clip.reshape(30, -1, 3).copy()   # 30 frames, landmarks, (x, y, z)
    mask = c != 0                        # keep empty (missing) landmarks at zero
    scale = np.random.uniform(0.9, 1.1)
    shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 3))
    noise = np.random.normal(0, 0.005, c.shape)
    c = (c * scale + shift + noise) * mask
    return c.reshape(clip.shape)

COPIES = 5   # extra versions made from each training clip
aug_X, aug_y = [], []
for clip, label in zip(X_train, y_train):
    for _ in range(COPIES):
        aug_X.append(augment(clip))
        aug_y.append(label)

X_train = np.concatenate([X_train, np.array(aug_X)])
y_train = np.concatenate([y_train, np.array(aug_y)])
print(f"[INFO] Training clips after augmentation: {len(X_train)}")
# -------------------------------------------------

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), callbacks=[early_stop])

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[RESULT] Test accuracy for {CATEGORY}: {acc*100:.1f}%")

os.makedirs('models', exist_ok=True)
model.save(f'models/{CATEGORY}.keras')
np.save(f'models/{CATEGORY}_actions.npy', np.array(actions))
print(f"[INFO] Saved models/{CATEGORY}.keras")