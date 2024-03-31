
from PIL import Image
import os

# make gifs loop forever
target_dir = "results/gifs/"
gifs = os.listdir(target_dir)
print(gifs)

for item in gifs:
    print(item)
    if ".gif" in item:
        g = Image.open(target_dir + item)
        g.save("results/" + item, save_all=True, loop=0)

print("done!")
