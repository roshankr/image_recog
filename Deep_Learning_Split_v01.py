import numpy as np
np.random.seed(2016)
import os
import glob
import sys
import warnings
import platform
import random
import time as tm
import pandas as pd
from shutil import copyfile

##########################################################################
# split the train and test data
##########################################################################


def split_data():

    sub_folder = []
    for folder in glob.glob(os.path.join(file_path, "full", "*/")):
        sub_folder.append(os.path.basename(os.path.dirname(folder)))

    if not(os.path.isdir(os.path.join(file_path, "train"))):
        os.makedirs(os.path.join(file_path, "train"))

    if not(os.path.isdir(os.path.join(file_path, "test"))):
        os.makedirs(os.path.join(file_path, "test"))

    for tf in sub_folder:

        path2 = os.path.join(file_path, "full", tf)
        path = os.path.join(path2, '*.pgm')

        try:
            os.makedirs(os.path.join(file_path, "train", tf))
        except BaseException:
            for f in glob.glob(os.path.join(file_path, "train", tf, "*")):
                os.remove(f)

        try:
            os.makedirs(os.path.join(file_path, "test", tf))
        except BaseException:
            for f in glob.glob(os.path.join(file_path, "test", tf, "*")):
                os.remove(f)

        files = glob.glob(path)
        random.shuffle(files)

        for fl in files[0:(train_ratio / 10)]:
            target = os.path.join(file_path, 'train', tf, os.path.basename(fl))
            copyfile(fl, target)

        for fl in files[(train_ratio / 10):len(files)]:
            target = os.path.join(file_path, 'test', tf, os.path.basename(fl))
            copyfile(fl, target)

##########################################################################
#Main module                                                                                                           #
##########################################################################


def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    global train_ratio

    warnings.filterwarnings("ignore")

    global file_path

    if(platform.system() == "Windows"):

        file_path = 'C:\\Python\\Others\\data\\att_faces'

    else:
        file_path = '/mnt/hgfs/Python/Others/data/att_faces/'

    try:
        train_ratio = int(argv[1])
    except BaseException:
        train_ratio = 60

    print("Train ratio is " + str(train_ratio))

    split_data()


##########################################################################
#Main program starts here                                                                                              #
##########################################################################
if __name__ == "__main__":
    main(sys.argv)
