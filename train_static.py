import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from static_gnn import HandGCN

d = np.load('static_features.npz')
X_all, labels_all = d['X'], d['y']


def mirror(f):
    a = f[:63].reshape(21, 3).copy()
    b = f[63:].reshape(21, 3).copy()
    a[:, 0] *= -1
    b[:, 0] *= -1
    if b.any():
        return np.concatenate([b.flatten(), a.flatten()])
    return np.concatenate([a.flatten(), b.flatten()])


def build_model(n):
    inp = layers.Input(shape=(126,))
    x = layers.Reshape((42, 3))(inp)
    x = HandGCN(32)(x)
    x = HandGCN(64)(x)
    x = HandGCN(64)(x)                       # 3-layer GNN on the hand graph
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n, activation='softmax')(x)
    return models.Model(inp, out)


def train(kind, keep):
    mask = np.array([keep(l) for l in labels_all])
    X, labels = X_all[mask], labels_all[mask]
    classes = sorted(set(labels))
    idx = np.array([classes.index(l) for l in labels])
    y = to_categorical(idx, len(classes))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=idx)
    X_tr = np.concatenate([X_tr, np.array([mirror(f) for f in X_tr])])
    y_tr = np.concatenate([y_tr, y_tr])

    model = build_model(len(classes))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_te, y_te),
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
              verbose=0)
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f'[RESULT] {kind} (GNN) test accuracy: {acc*100:.1f}%')
    os.makedirs('static_model', exist_ok=True)
    model.save(f'static_model/{kind}_gnn.keras')
    np.save(f'static_model/{kind}_gnn_actions.npy', np.array(classes))


train('Numbers', lambda l: l.isdigit())
train('Letters', lambda l: l.isalpha())
print('[INFO] Saved Numbers_gnn and Letters_gnn in static_model/')