import os, re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from train_gcn_layer import GCNLayer

DATA_PATH = 'MP_Data'
MIN_CLIPS = 4          # words with fewer clips are skipped

# ---- 1. Group word folders by category ----
GROUPS = {}
for d in sorted(os.listdir(DATA_PATH)):
    if not os.path.isdir(os.path.join(DATA_PATH, d)):
        continue
    m = re.match(r'^([A-Za-z_]+?)_\d+\.', d)
    if m:
        cat = m.group(1)
    elif re.match(r'^\d+\.', d):
        cat = 'Adjectives'
    else:
        continue                     # old webcam folders (hello/iloveyou/thanks)
    GROUPS.setdefault(cat, []).append(d)

# ---- 2. Split Days_and_Time into 3 smaller groups ----
SPLIT = {
    'Days_and_Time': {
        'Weekdays': ['66.', '67.', '68.', '69.', '70.', '71.'],
        'TimeOfDay': ['82.', '83.', '84.', '85.'],
        'TimeUnits': ['74.', '75.', '76.', '77.', '78.', '79.', '81.', '86.'],
    }
}
for cat, parts in SPLIT.items():
    words = GROUPS.pop(cat, [])
    for new, ids in parts.items():
        GROUPS[new] = [w for w in words if any(f'_{i}' in w for i in ids)]

# ---- 3. Merge tiny categories ----
MERGE = {'Everyday': ['Clothes', 'Colours', 'Electronics'],
         'Misc': ['Home', 'People', 'Means_of_Transportation']}
for new, olds in MERGE.items():
    GROUPS[new] = [w for o in olds for w in GROUPS.pop(o, [])]

print('[INFO] Categories to train:')
for c, w in GROUPS.items():
    print(f'   {c}: {len(w)} word folders')


def build_model(n):
    inp = layers.Input(shape=(30, 150))
    x = layers.Reshape((30, 50, 3))(inp)
    x = GCNLayer(32)(x)
    x = GCNLayer(64)(x)
    x = GCNLayer(64)(x)                         # 3-layer GCN
    x = layers.Reshape((30, 50 * 64))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n, activation='softmax')(x)
    return models.Model(inp, out)


def augment(clip):
    c = clip.reshape(30, -1, 3).copy()
    mask = c != 0
    c = (c * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05, (1, 1, 3))
         + np.random.normal(0, 0.005, c.shape)) * mask
    return c.reshape(clip.shape)


os.makedirs('models', exist_ok=True)
summary = []

for cat, words in GROUPS.items():
    print(f'\n[INFO] Training {cat} ...')
    seqs, labels, used = [], [], []
    for w in words:
        wp = os.path.join(DATA_PATH, w)
        clips = []
        for s in os.listdir(wp):
            try:
                clips.append([np.load(os.path.join(wp, s, f'{f}.npy')) for f in range(30)])
            except Exception:
                pass
        if len(clips) >= MIN_CLIPS:
            used.append(w)
            seqs += clips
            labels += [len(used) - 1] * len(clips)
        else:
            print(f'   [skip] {w}: only {len(clips)} clips')

    if len(used) < 2:
        print(f'[SKIP] {cat}: not enough words')
        continue

    X = np.array(seqs)
    y = to_categorical(labels, len(used))
    counts = np.bincount(labels)
    strat = labels if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    aug = [augment(c) for c in X_tr for _ in range(5)]
    aug_y = [l for l in y_tr for _ in range(5)]
    X_tr = np.concatenate([X_tr, np.array(aug)])
    y_tr = np.concatenate([y_tr, np.array(aug_y)])

    model = build_model(len(used))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_tr, y_tr, epochs=150, verbose=0, validation_data=(X_te, y_te),
              callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])
    acc = model.evaluate(X_te, y_te, verbose=0)[1]
    print(f'[RESULT] {cat}: {len(used)} words, {len(X)} clips, test accuracy {acc*100:.1f}%')
    summary.append((cat, len(used), len(X), acc))

    model.save(f'models/{cat}_gcn.keras')
    np.save(f'models/{cat}_gcn_actions.npy', np.array(used))

print('\n===== SUMMARY =====')
for cat, n, c, a in summary:
    print(f'{cat:15s} words={n:3d} clips={c:4d} acc={a*100:.1f}%')