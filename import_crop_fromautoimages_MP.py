import cv2
import os
from datetime import date
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import glob
import shutil

today = date.today()
now = today.strftime("%d%m%Y")
# original images are 1292x964

cwd = os.getcwd()
snapshots_location = "/dls/i23/data/2022/cm31108-3/Sample_Loading_Snapshots/ECAM_6"
ON_folders = ["After_sample_load", "Pin_gripper_on_gonio"]  # "Gripper_approach_gonio"
OFF_folders = [
    "Before_sample_load",
    "Gripper_approach_hotel",
]  # use similar number of images
folder_list = ["dark", "light", "pinon", "pinoff"]
#     "Gripper_gripping_hotel_pin",
#     "Gripper_retracted",
#     "Hotel_in_place",
#     "Hotel_rotated_away_gripping_pin",
#     "Pin_above_hotel_position",
#     "Pin_place_in_block",
# ]
path = os.path.join(cwd, f"goniopin_auto_{now}")


def croppit(filein, folderout):
    img = cv2.imread(filein)
    if img is None:
        print(f"Failed to load image: {filein}")
        return
    cropped_image = img[:, :]
    _, filename = os.path.split(filein)
    cv2.imwrite(os.path.join(folderout, filename), cropped_image)


def changeBrightness(imageIn, dirOut, factor):
    if factor < 1:
        ending = "d"
    elif factor > 1:
        ending = "l"
    with Image.open(imageIn) as img:
        enhancer = ImageEnhance.Brightness(img)
        enhanced = enhancer.enhance(factor)
        enhanced.save(os.path.join(dirOut, ending + os.path.basename(imageIn)))


def calculateAverageColor(image):
    np_image = np.array(image)
    avg_color = np.mean(np_image, axis=(0, 1)).astype(int)
    return tuple(avg_color)


def augmentSave(imageIn, num_augmented=20):
    image = Image.open(imageIn)
    avg_colour = calculateAverageColor(image)

    for i in range(num_augmented):
        augmented_image = image.copy()

        # Apply random rotation
        angle = random.uniform(-1, 1)
        augmented_image = augmented_image.rotate(
            angle, resample=Image.BICUBIC, fillcolor=avg_colour
        )

        # Apply random translation
        max_dx = 0.04 * augmented_image.size[0]
        max_dy = 0.04 * augmented_image.size[1]
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)
        augmented_image = augmented_image.transform(
            augmented_image.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BICUBIC,
            fillcolor=avg_colour,
        )

        # Apply random brightness
        enhancer = ImageEnhance.Brightness(augmented_image)
        augmented_image = enhancer.enhance(random.uniform(0.6, 1.4))

        # Apply random contrast
        enhancer = ImageEnhance.Contrast(augmented_image)
        augmented_image = enhancer.enhance(random.uniform(0.6, 1.4))

        augmented_image.save(
            os.path.join(
                os.path.dirname(imageIn),
                f"{os.path.basename(imageIn).split('.')[0]}_aug_{i}.jpg",
            )
        )


def processImages():
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        futures = []
        for folder in folder_list:
            folder_path = os.path.join(path, folder)
            for image_name in os.listdir(folder_path):
                if image_name.endswith("jpg"):
                    image_path = os.path.join(folder_path, image_name)
                    futures.append(executor.submit(augmentSave, image_path))
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Augmenting images"
        ):
            future.result()


def generateDarkLight():
    darkDir = os.path.join(path, "dark")
    lightDir = os.path.join(path, "light")
    pinOnDir = os.path.join(path, "pinon")
    pinOffDir = os.path.join(path, "pinoff")
    pinOnImages = [
        file
        for file in os.listdir(pinOnDir)
        if os.path.isfile(os.path.join(pinOnDir, file))
    ]
    pinOffImages = [
        file
        for file in os.listdir(pinOffDir)
        if os.path.isfile(os.path.join(pinOffDir, file))
    ]
    pinOnimageSelect = random.sample(pinOnImages, int(len(pinOnImages) * 0.5))
    pinOffimageSelect = random.sample(pinOffImages, int(len(pinOffImages) * 0.5))

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        futures = []
        for imageName in pinOnimageSelect:
            imagePath = os.path.join(pinOnDir, imageName)
            futures.append(
                executor.submit(
                    changeBrightness, imagePath, darkDir, random.uniform(0.0001, 0.0008)
                )
            )
            futures.append(
                executor.submit(
                    changeBrightness, imagePath, lightDir, random.uniform(5.1, 6)
                )
            )
        for imageName in pinOffimageSelect:
            imagePath = os.path.join(pinOffDir, imageName)
            futures.append(
                executor.submit(
                    changeBrightness, imagePath, darkDir, random.uniform(0.0001, 0.0006)
                )
            )
            futures.append(
                executor.submit(
                    changeBrightness, imagePath, lightDir, random.uniform(4.5, 6)
                )
            )
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing light and darks"
        ):
            future.result()


def moveImagesToTest(path=path, percentage=0.1):
    testdir = os.path.join(os.getcwd(), f"test_{now}")
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    for folder in folder_list:
        classdir = os.path.join(path, folder)
        testclassdir = os.path.join(testdir, folder)

        if not os.path.exists(testclassdir):
            os.makedirs(testclassdir)

        images = glob.glob(os.path.join(classdir, "*.jpg"))
        numImg = int(len(images) * percentage)
        imgMov = random.sample(images, numImg)

        for image in imgMov:
            shutil.move(image, testclassdir)


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

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        futures = []
        for pinon_image_dir in ON_folders:
            searchdir = os.path.join(snapshots_location, "pin_ON", pinon_image_dir)
            files = [f for f in os.listdir(searchdir) if f.endswith("jpg")]
            for file in files:
                image = os.path.join(searchdir, file)
                futures.append(
                    executor.submit(croppit, image, os.path.join(path, "pinon"))
                )

        for pinoff_image_dir in OFF_folders:
            searchdir = os.path.join(snapshots_location, "pin_OFF", pinoff_image_dir)
            files = [f for f in os.listdir(searchdir) if f.endswith("jpg")]
            for file in files:
                image = os.path.join(searchdir, file)
                futures.append(
                    executor.submit(croppit, image, os.path.join(path, "pinoff"))
                )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing images"
        ):
            future.result()


if __name__ == "__main__":
    run()
    generateDarkLight()
    moveImagesToTest()
    processImages()
