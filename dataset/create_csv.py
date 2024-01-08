from PIL import Image
import numpy as np
import sys
import os
import csv


# default format can be changed as needed
def createFileList(myDir, format=".png"):
    fileList = []
    labels = []
    names = []
    keywords = {
        "z": 0,
        "o": 1,
        "t": 2,
        "r": 3,
        "f": 4,
        "v": 5,
        "x": 6,
        "s": 7,
        "e": 8,
        "i": 9,
    }  # keys and values to be changed as needed
    for root, dirs, files in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
            for keyword in keywords:
                if keyword in name:
                    labels.append(keywords[keyword])
                else:
                    continue
            names.append(name)
    return fileList, labels, names


# load the original image
myFileList, labels, names = createFileList("./numbers")

i = 0
for file in myFileList:
    print(file)
    img_file = Image.open(file)

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode
    # Make image Greyscale
    img_grey = img_file.convert("L")

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=int).reshape((width, height))
    value = value.flatten()

    value = np.append(value, labels[i])
    i += 1

    # print(value)
    with open("numbers.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ", lineterminator="\n")
        writer.writerow(value)
