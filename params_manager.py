import json
import os


JSON_FILE = './params.json'

class ParamsManager(object):
    
    def load(self) -> dict:
        if not os.path.exists(JSON_FILE):
            self.save({})
            
        with open(JSON_FILE, 'r') as f:
            return json.load(f)

    def save(self, params: dict):
        with open(JSON_FILE, 'w') as f:
            json.dump(params, f)
