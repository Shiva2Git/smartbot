import json

# Example JSON object
json_data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Convert JSON object to raw text
raw_text = ' '.join(f"{key} {value}" for key, value in json_data.items())

print(type(raw_text))
