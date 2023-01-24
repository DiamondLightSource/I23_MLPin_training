import os
from pathlib import Path
from PIL import Image, ImageEnhance

def dim(file):
    im = Image.open(file)
    out = Path(file).stem
    enhancer = ImageEnhance.Brightness(im)
    factor = 5.0
    im_output = enhancer.enhance(factor)
    im_output.save(os.path.join("/dls/science/groups/i23/scripts/chris/I23_MLPin_training/goniopin_auto_12012023/light", str(out) + "_2.jpg"))

if __name__=="__main__":
    path = os.getcwd()
    if os.path.exists(path):
        print(f"{path} exists")
    else:
        pass
    for file in os.listdir(path):
        print(file)
        dim(file)
