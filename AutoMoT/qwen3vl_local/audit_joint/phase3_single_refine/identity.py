import hashlib
import json

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def mass(target,count):
    if not target['valid']:return [0.]*count
    return list(target['target']) if 'target' in target else [float(i==target['token_id']) for i in range(count)]
