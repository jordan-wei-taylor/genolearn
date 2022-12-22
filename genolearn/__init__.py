"""
genolearn doc

"""

import os

__version__ = '0.0.9'

ls = os.listdir()
wd = True if '.genolearn' in ls else False

path = os.path.join(os.path.dirname(__file__), 'wd')
if wd:
    working_directory = os.path.abspath('.')
    with open(path, 'w') as f:
        f.write(working_directory)
elif os.path.exists(path):
    with open(path) as f:
        working_directory = f.read()
else:
    working_directory = None

ls = os.listdir(working_directory if working_directory and os.path.exists(working_directory) else '.')

def listdir(path = '.'):
    if working_directory and os.path.exists(working_directory):
        path = os.path.join(working_directory, path)
        return os.listdir(path) if os.path.exists(path) else []
    return []

def get_active():
    try:
        if working_directory:
            path = os.path.join(working_directory, '.genolearn')
            if os.path.exists(path):
                import json
                with open(path) as f:
                    return json.load(f)
    except:
        ...