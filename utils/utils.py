import json

def read_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)

def write_json(content, fpath):
    with open(fpath, 'w') as f:
        json.dump(content, f, indent=4)
