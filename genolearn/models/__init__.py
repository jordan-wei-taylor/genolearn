from . import classification

import joblib
import os

root = 'models'

def set_dir(path):
    """test"""
    global root
    root = path

def get_dir():
    global root
    return root

def save(model, path, overwrite = False):
    global root
    os.makedirs(root, exist_ok = True)
    full_path = os.path.join(root, path)
    if os.path.exists(full_path):
        if overwrite:
            raise Exception(f'"{full_path}" already exists!')
    
    joblib.dump(model, full_path)

def load(path):
    global root
    full_path = os.path.join(root, path)
    if os.path.exists(full_path):
        return joblib.load(path)
    raise Exception(f'"{full_path}" does not exist!')