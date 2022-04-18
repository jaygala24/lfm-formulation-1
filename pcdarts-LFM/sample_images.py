import os
import shutil
import glob
import random

classes = os.listdir("../data/imagenet/train/")
new_classes = []
for item in classes:
    if os.path.isdir(os.path.join("../data/imagenet/train/", item)):
        new_classes.append(item)
assert len(new_classes) == 1000

for name in new_classes:
    files = glob.glob("../data/imagenet/train/" + name + "/*.JPEG")
    number = len(files)
    to_be_moved = random.sample(files, int(number * 0.08))
    print(f"[train] copying samples from {name} ...")

    for f in enumerate(to_be_moved, 1):
        dest = os.path.join("../data/imagenet_sampled/train/", name)
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)

for name in new_classes:
    files = glob.glob("../data/imagenet_sampled/train/" + name + "/*.JPEG")
    number = len(files)
    to_be_moved = random.sample(files, int(number * 0.025))
    print(f"[val] copying samples from {name} ...")

    for f in enumerate(to_be_moved, 1):
        dest = os.path.join("../data/imagenet_sampled/val/", name)
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.move(f[1], dest)
