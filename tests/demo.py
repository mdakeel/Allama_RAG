import json

with open("data/transcripts/_uS8c_Vq5P8.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data.keys())          # top-level keys
print(data)                 # sample content
