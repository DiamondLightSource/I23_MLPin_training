import cv2
import os

dir_orig = input("Path to files: ")
path = os.getcwd()
dir = os.path.join(path, "goniopin", dir_orig)
croppedpath = os.path.join(path, "goniopin", "cropped", dir_orig)


def showimg(toshow):
    cv2.imshow("image", toshow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def croppit(filein):
    img = cv2.imread(filein)
    cropped_image = img[400:750, 610:770]
    outfile = os.path.join(croppedpath, os.path.basename(filein))
    cv2.imwrite(outfile, cropped_image)


print(dir)
for file in os.listdir(dir):
    if file.endswith("jpg"):
        print(file)
        file = os.path.join(dir, file)
        croppit(file)
