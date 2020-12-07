from PIL import Image
import torch
import torchvision

import os

ROTATE_RANGE = [-10, 10]
SCALE_RANGE = [1.1, 1.2]
EXPAND = 9

randomize = torchvision.transforms.RandomAffine(ROTATE_RANGE, scale=SCALE_RANGE, fillcolor=255)

n_classes = 100
n_originals = 5
root_path = r'BMP600'

if __name__ == "__main__":
    for c in range(n_classes):
        for o in range(n_originals):
            path = "{}/{:03d}/{:03d}_{:1d}.bmp".format(root_path, c+1, c+1, o+1)
            img = Image.open(path)
            for r in range(EXPAND):
                new_path = "{}/{:03d}/{:03d}_{:1d}_ex{:02d}.bmp".format(root_path, c+1, c+1, o+1, r+1)
                new_img = randomize(img)
                new_img.save(new_path, 'BMP')
        print("finished class {:03d}".format(c+1))

# if __name__ == "__main__":
#     for c in range(1, n_classes):
#         for o in range(n_originals):
#             for e in range(EXPAND):
#                 cmd = "del \"{}\{:03d}\{:03d}_{}_{:02d}.bmp\"".format(
#                     root_path, c+1, c+1, o+1, e+1)
#                 print("running: {}".format(cmd))
#                 os.system(cmd)
