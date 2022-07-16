import cv2
import os

path = os.getcwd()
croppedpath = os.path.join(path, "cropped", "pin_off")

def showimg(toshow):    
    cv2.imshow("image", toshow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def croppit(filein):
    img = cv2.imread(filein)
    cropped_image = img[400:650, 550:800]
    outfile = os.path.join(croppedpath, os.path.basename(filein))
    cv2.imwrite(outfile, cropped_image)
    
dir = input("Path to files: ")
dir = os.path.join(path, dir)
print(dir)
for file in os.listdir(dir):
    if file.endswith("jpg"):
        print(file)
        file = os.path.join(dir, file)
        croppit(file)

