import os
import numpy as np
from skimage import io
from skimage.transform import resize

RESIZED_SHAPE = (64,64,3)
IMG_DIR = "./demo_imgs"

filenames = os.listdir(IMG_DIR)

X = np.array([])

for i,filename in enumerate(filenames):
    img = io.imread(os.path.join(IMG_DIR,filename))
    if img.shape != RESIZED_SHAPE:
        img = resize(img,RESIZED_SHAPE,preserve_range=True)
    img = img.astype("float64")
    img = img.reshape(1,-1).T
    if i == 0:
        X = img
    else:
        X = np.concatenate((X,img),1)
    
X /= 255.0
    
np.savetxt(f"fnames",np.array([filenames]).reshape(-1,1),fmt="%s")
np.savetxt(f"demo",X)
