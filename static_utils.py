import numpy as np

def hands_to_features(hand_list):
    """hand_list = list of (21,3) arrays. Returns 126 numbers (2 hands, padded)."""
    hs = []
    for h in hand_list:
        h = np.array(h, dtype=np.float32)
        wrist_x = float(h[0, 0])
        h = h - h[0]                      # wrist becomes the origin
        m = np.abs(h).max()
        if m > 0:
            h = h / m                     # remove hand size / distance
        hs.append((wrist_x, h.flatten()))
    hs.sort(key=lambda t: t[0])           # left-most hand first
    out = [h for _, h in hs[:2]]
    while len(out) < 2:
        out.append(np.zeros(63, dtype=np.float32))
    return np.concatenate(out)

def landmarks_from_result(result):
    if not result.multi_hand_landmarks:
        return []
    return [[(l.x, l.y, l.z) for l in hl.landmark] for hl in result.multi_hand_landmarks]