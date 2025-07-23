import cv2
import os
from datetime import date
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import glob
import shutil

today = date.today()
now = today.strftime("%d%m%Y")
tmpdir = "/dls/tmp/vwg85559"
# original images are 1292x964

cwd = os.getcwd()
snapshots_location = "/dls/i23/data/2022/cm31108-3/Sample_Loading_Snapshots/ECAM_6"
ON_folders = ["Pin_gripper_on_gonio"]  #"After_sample_load"# "Gripper_approach_gonio"
OFF_folders = ["Before_sample_load"]
  # use similar number of images
folder_list = ["pinon", "pinoff"]
#     "Gripper_gripping_hotel_pin",
#     "Gripper_retracted",
#     "Hotel_in_place",
#     "Hotel_rotated_away_gripping_pin",
#     "Pin_above_hotel_position",
#     "Pin_place_in_block",
# ]
path = os.path.join(tmpdir, f"goniopin_auto_{now}_binary")


def croppit(filein, folderout):
    img = cv2.imread(filein)
    if img is None:
        print(f"Failed to load image: {filein}")
        return
    cropped_image = img[100:-100, 100:-100]
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


def augmentSave(imageIn, num_augmented=0):
    image = Image.open(imageIn)
    image_np = np.array(image)

    for i in range(num_augmented):
        augmented_image = image_np.copy()

        # Apply random rotation
        angle = random.uniform(-1, 1)
        M = cv2.getRotationMatrix2D(
            (augmented_image.shape[1] / 2, augmented_image.shape[0] / 2), angle, 1
        )
        augmented_image = cv2.warpAffine(
            augmented_image,
            M,
            (augmented_image.shape[1], augmented_image.shape[0]),
            borderMode=cv2.BORDER_REFLECT,
        )

        # Apply random translation
        max_dx = 0.1 * augmented_image.shape[1]
        max_dy = 0.1 * augmented_image.shape[0]
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        augmented_image = cv2.warpAffine(
            augmented_image,
            M,
            (augmented_image.shape[1], augmented_image.shape[0]),
            borderMode=cv2.BORDER_REFLECT,
        )

        # Convert back to PIL Image for brightness and contrast adjustments
        augmented_image = Image.fromarray(augmented_image)

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
            ),
            quality=95,
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


def moveImagesToTest(path=path, percentage=0.1):
    testdir = os.path.join(cwd, f"test_{now}_binary")
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

        for image in tqdm(imgMov, desc=f"Moving images from {folder}"):
            shutil.move(image, testclassdir)


def run():
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    for folder in ("pinoff", "pinon"):
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
    moveImagesToTest()
    processImages()
