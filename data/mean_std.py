from skimage import io
import numpy as np
import pandas as pd
import progressbar

data = pd.read_csv('train_list.csv')
path_arr = np.asarray(data['filepath'])

## mean: [0.48548178 0.48455666 0.46329196] std: [0.21904471 0.21578524 0.23359051]

count = 0
total_mean = np.zeros(3)
M2 = np.zeros(3)
for i in progressbar.progressbar(range(len(path_arr))):
    image = io.imread(str(path_arr[i]))[:, :, :3]/255.0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            count += 1
            val = image[x, y, :]
            delta = val - total_mean
            total_mean = total_mean +  delta/count
            delta2 = val - total_mean
            M2 = M2 + delta * delta2
print('mean:', total_mean, 'std:', np.sqrt(M2/(count - 1)))