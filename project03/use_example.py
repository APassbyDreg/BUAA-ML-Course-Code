from DiffModel import *
from PIL import Image

import json


model = torch.load(r"models\results\diff_model_v1_11-10_08-13.pkl")
ds = json.load(open(r"myds\post_res50_expanded.json"))


# case of using a img
img = Image.open(r"BMP600_test\001\001_6.bmp")
target = img2input(img)


# case of using a converted list:
# target = ds['valid'][0][0]


print(predict(ds['train'], target, model))
