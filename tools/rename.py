import os
import shutil

path = './ped2/training/'
files = os.listdir(path)
files.sort()
for filename in files:
    frames = os.listdir(path + filename)
    frames.sort()
    i=0
    for frame in frames:
        if frame.endswith(".tif"):
            src = os.path.join(path + filename + '/', frame)
            dst = os.path.join(path + filename + '/', "{:03d}.jpg".format(i))
            i = i + 1
            shutil.move(src, dst)
