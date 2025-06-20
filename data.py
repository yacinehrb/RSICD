import pandas as pd
from PIL import Image
import io
import os
import json

def convert_ndarray_to_list(obj):
    if isinstance(obj, (list, tuple)):
        return [convert_ndarray_to_list(i) for i in obj]
    elif hasattr(obj, 'tolist'):  # likely np.ndarray
        return obj.tolist()
    else:
        return obj

train=  r"C:\Users\yacine-abdelaziz.her\Remote sensing image captioning\RSICD\data\train-00000-of-00001.parquet"
test= r"C:\Users\yacine-abdelaziz.her\Remote sensing image captioning\RSICD\data\test-00000-of-00001.parquet"
valid= r"C:\Users\yacine-abdelaziz.her\Remote sensing image captioning\RSICD\data\valid-00000-of-00001.parquet"

df = pd.read_parquet(valid)

# Directory to save images
output_dir = r"RSICD\valid\valid_images"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to hold image filenames and their captions
image_captions = {}
data_list = []
for idx, row in df.iterrows():
    filename = row["filename"].split("/")[-1]
    image_bytes = row["image"]["bytes"]
    captions = row['captions']

    image = Image.open(io.BytesIO(image_bytes))
    image_save_path = os.path.join(output_dir, filename)
    image.save(image_save_path)

    entry = {
        "image_path": image_save_path.replace("\\", "/"),
        "captions": convert_ndarray_to_list(captions)
    }
    data_list.append(entry)

    if idx % 100 == 0:
        print(f"Processed {idx+1}/{len(df)} images...")

json_output_path = r"RSICD\valid_captions.json"
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"Saved captions JSON to {json_output_path}")


