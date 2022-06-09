import json
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)

def write_json(content, fpath):
    with open(fpath, 'w') as f:
        json.dump(content, f, indent=4)

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])