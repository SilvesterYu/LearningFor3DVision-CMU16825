
from PIL import Image
import os

# make gifs loop forever
target_dir = "results/vox_Sel/"
gifs = os.listdir(target_dir)
print(gifs)

for item in gifs:
    print(item)
    if ".gif" in item:
        g = Image.open(target_dir + item)
        g.save("results_loop/" + item, save_all=True, loop=0)

print("done!")
