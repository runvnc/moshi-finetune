import json
import os
import uuid

file_path = "data/custom_dataset/raw_transcripts.json"
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    migrated = False
    new_data = []
    for item in data:
        if isinstance(item, list):
            new_data.append({"id": str(uuid.uuid4()), "dialogue": item})
            migrated = True
        else:
            new_data.append(item)
            
    if migrated:
        with open(file_path, "w") as f:
            json.dump(new_data, f, indent=2)
        print("Migrated raw_transcripts.json to use UUIDs.")
