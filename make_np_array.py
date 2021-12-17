import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

df = pd.read_json("set_10000.json")[:10]
df = df[["name","token_id"]]

img_arr = np.zeros((df.index.size, 128, 128, 3), dtype=np.float32)

for ind in df.index:
    print(df["name"][ind], f"{ind}/{df.index.size}")
    image_path = "out/" + str(df["token_id"][ind]) + ".png"
    image = Image.open(image_path)
    data = asarray(image)
    data = (data - 127.5) / 127.5
    img_arr[ind] = data
    
img_arr = img_arr.reshape(df.index.size, 128, 128, 3).astype(dtype=np.float32)
    
print(img_arr)
print(img_arr.shape)
print(img_arr.size)

with open("img_arr.npy", "wb") as f:
    print("save")
    np.save(f, img_arr)
