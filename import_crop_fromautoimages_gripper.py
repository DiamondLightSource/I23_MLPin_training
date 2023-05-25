import cv2
import os
from datetime import date

today = date.today()
now = today.strftime("%d%m%Y")


def croppit(filein, folderout):
    img = cv2.imread(filein)
    cropped_image = img[300:600, 610:770]
    _, filename = os.path.split(filein)
    cv2.imwrite(os.path.join(folderout, filename), cropped_image)


cwd = os.getcwd()
snapshots_location = "/dls/i23/data/2022/cm31108-3/Sample_Loading_Snapshots/ECAM_6"
open_folders = ["Gripper_approach_gonio"]
haspin_folders = ["Gonio_approach_with_pin"]
closed_folders = [""]
path = os.path.join(cwd, f"gripperstatus_auto_{now}")


def run():
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    if os.path.exists(os.path.join(path, "gripperOpen")):
        pass
    else:
        os.mkdir(os.path.join(path, "gripperOpen"))

    if os.path.exists(os.path.join(path, "gripperHasPin")):
        pass
    else:
        os.mkdir(os.path.join(path, "gripperHasPin"))

    if os.path.exists(os.path.join(path, "gripperClosed")):
        pass
    else:
        os.mkdir(os.path.join(path, "gripperClosed"))

    for gripperOpenImgDir in open_folders:
        searchdir = os.path.join(snapshots_location, "pin_ON", gripperOpenImgDir)
        for file in os.listdir(searchdir):
            if file.endswith("jpg"):
                image = os.path.join(searchdir, file)
                croppit(image, os.path.join(path, "gripperOpen"))

    for gripperHasPinImgDir in haspin_folders:
        searchdir = os.path.join(snapshots_location, "pin_OFF", gripperHasPinImgDir)
        for file in os.listdir(searchdir):
            if file.endswith("jpg"):
                image = os.path.join(searchdir, file)
                croppit(image, os.path.join(path, "gripperHasPin"))


if __name__ == "__main__":
    run()
