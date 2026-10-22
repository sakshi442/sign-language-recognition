import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

HAND = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
BODY = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7)]
edges = list(BODY)
edges += [(a + 8, b + 8) for a, b in HAND]
edges += [(a + 29, b + 29) for a, b in HAND]
edges += [(4, 8), (5, 29)]

A = np.eye(50, dtype=np.float32)
for a, b in edges:
    A[a, b] = A[b, a] = 1
D = np.diag(1.0 / np.sqrt(A.sum(1)))
A_NORM = (D @ A @ D).astype(np.float32)


@tf.keras.utils.register_keras_serializable()
class GCNLayer(layers.Layer):
    def __init__(self, units, **kw):
        super().__init__(**kw)
        self.units = units
        self.dense = layers.Dense(units, activation='relu')
        self.adj = tf.constant(A_NORM)

    def call(self, x):
        x = tf.einsum('ij,btjf->btif', self.adj, x)
        return self.dense(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg