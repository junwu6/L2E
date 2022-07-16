import scipy.io as sio
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


if __name__ == '__main__':
    root = '../../data/caltran/'
    data = sio.loadmat(root + 'caltran_dataset_labels.mat')
    image_names = data['names']
    image_labels = data['labels']

    for d in range(4, 16):
        directory_p = '../../data/caltran_continuous/' + str(d) + '/positive'
        directory_n = '../../data/caltran_continuous/' + str(d) + '/negative'
        if not os.path.exists(directory_p):
            os.makedirs(directory_p)
            os.makedirs(directory_n)
        for i in range(len(image_names)):
            name = image_names[i][0][0]
            label = image_labels[0][i]

            if int(name[8:10]) == d and label == 1:
                img_PIL = Image.open(r'{}.jpg'.format(root + name))
                img_PIL.save(r'{}.jpg'.format(directory_p + '/' + name))
            elif int(name[8:10]) == d and label == -1:
                img_PIL = Image.open(r'{}.jpg'.format(root + name))
                img_PIL.save(r'{}.jpg'.format(directory_n + '/' + name))
