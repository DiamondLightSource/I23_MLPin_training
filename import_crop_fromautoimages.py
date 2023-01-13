import cv2
import os
from datetime import date

today = date.today()
now = today.strftime("%d%m%Y")

def croppit(filein, folderout):
    img = cv2.imread(filein)
    cropped_image = img[400:700, 610:770]
    _, filename = os.path.split(filein)
    cv2.imwrite(os.path.join(folderout, filename), cropped_image)


cwd = os.getcwd()
snapshots_location = "/dls/i23/data/2022/cm31108-3/Sample_Loading_Snapshots/ECAM_6"
ON_folders = ["After_sample_load", "Pin_gripper_on_gonio"]
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

def run():
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    if os.path.exists(os.path.join(path, "pinoff")):
        pass
    else:
        os.mkdir(os.path.join(path, "pinoff"))

    if os.path.exists(os.path.join(path, "pinon")):
        pass
    else:
        os.mkdir(os.path.join(path, "pinon"))

    for pinon_image_dir in ON_folders:
        searchdir = os.path.join(snapshots_location, "pin_ON", pinon_image_dir)
        for file in os.listdir(searchdir):
            if file.endswith("jpg"):
                image = os.path.join(searchdir, file)
                croppit(image, os.path.join(path, "pinon"))

    for pinon_image_dir in OFF_folders:
        searchdir = os.path.join(snapshots_location, "pin_OFF", pinon_image_dir)
        for file in os.listdir(searchdir):
            if file.endswith("jpg"):
                image = os.path.join(searchdir, file)
                croppit(image, os.path.join(path, "pinoff"))

if __name__=="__main__":
    run()