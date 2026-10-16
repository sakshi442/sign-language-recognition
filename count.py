import os

DATA_PATH = 'MP_Data'
for action in sorted(os.listdir(DATA_PATH)):
    path = os.path.join(DATA_PATH, action)
    if os.path.isdir(path):
        print(action, len(os.listdir(path)))