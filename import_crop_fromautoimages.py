import cv2
import os
from datetime import date
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

today = date.today()
now = today.strftime("%d%m%Y")
# original images are 1292x964

def croppit(filein, folderout):
    img = cv2.imread(filein)
    cropped_image = img[100:900, 200:1000]
    _, filename = os.path.split(filein)
    cv2.imwrite(os.path.join(folderout, filename), cropped_image)


cwd = os.getcwd()
snapshots_location = "/dls/i23/data/2022/cm31108-3/Sample_Loading_Snapshots/ECAM_6"
ON_folders = ["After_sample_load", "Pin_gripper_on_gonio", "Gripper_approach_gonio"]
OFF_folders = [
    "Before_sample_load",
    "Gripper_approach_hotel",
    "Gripper_gripping_hotel_pin",
    "Gripper_retracted",
    "Hotel_in_place",
    "Hotel_rotated_away_gripping_pin",
    "Pin_above_hotel_position",
    "Pin_place_in_block",
]
path = os.path.join(cwd, f"goniopin_auto_{now}")

def changeBrightness(imageIn, dirOut, factor):
    if factor < 1:
        ending = "d"
    elif factor > 1:
        ending = "l"
    with Image.open(imageIn) as img:
        enhancer = ImageEnhance.Brightness(img)
        enhanced = enhancer.enhance(factor)
        enhanced.save(os.path.join(dirOut, ending + os.path.basename(imageIn)))

def generateDarkLight():
    darkDir = os.path.join(path, "dark")
    lightDir = os.path.join(path, "light")
    pinOnDir = os.path.join(path, "pinon")
    pinOffDir = os.path.join(path, "pinoff")
    pinOnImages = [file for file in os.listdir(pinOnDir) if os.path.isfile(os.path.join(pinOnDir, file))]
    pinOffImages = [file for file in os.listdir(pinOffDir) if os.path.isfile(os.path.join(pinOffDir, file))]
    pinOnimageSelect = random.sample(pinOnImages, int(len(pinOnImages) * 0.5)) 
    pinOffimageSelect = random.sample(pinOffImages, int(len(pinOffImages) * 0.5))
    for imageName in tqdm(pinOnimageSelect, desc="Processing pinon light and darks"):
        imagePath = os.path.join(pinOnDir, imageName)
        changeBrightness(imagePath, darkDir, 0.03)
        changeBrightness(imagePath, lightDir, 4.3)
    for imageName in tqdm(pinOffimageSelect, desc="Processing pinoff light and darks"):
        imagePath = os.path.join(pinOffDir, imageName)
        changeBrightness(imagePath, darkDir, 0.03)
        changeBrightness(imagePath, lightDir, 4.3)

def run():
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    for folder in ("pinoff", "pinon", "dark", "light"):
        if os.path.exists(os.path.join(path, folder)):
            pass
        else:
            os.mkdir(os.path.join(path, folder))

    # for pinon_image_dir in ON_folders:
    #     searchdir = os.path.join(snapshots_location, "pin_ON", pinon_image_dir)
    #     files = [f for f in os.listdir(searchdir) if f.endswith("jpg")]
    #     for file in tqdm(files, desc=f"Processing {pinon_image_dir}"):
    #         image = os.path.join(searchdir, file)
    #         croppit(image, os.path.join(path, "pinon"))

    # for pinoff_image_dir in OFF_folders:
    #     searchdir = os.path.join(snapshots_location, "pin_OFF", pinoff_image_dir)
    #     files = [f for f in os.listdir(searchdir) if f.endswith("jpg")]
    #     for file in tqdm(files, desc=f"Processing {pinoff_image_dir}"):
    #         image = os.path.join(searchdir, file)
    #         croppit(image, os.path.join(path, "pinoff"))


if __name__ == "__main__":
    run()
    generateDarkLight()