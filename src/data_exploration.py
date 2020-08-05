import mrcfile
import numpy as np 
import json

tomo_path = '../data/dataset/SNR003/1bxn/subtomogram_mrc/tomotarget0.mrc'
tomo_label_path = '../data/dataset/SNR003/1bxn/json_label/target0.json'

# read tomogram
f = mrcfile.open(tomo_path)
tomo_arr = np.asarray(f.data)
print(tomo_arr.shape)

# read label
with open(tomo_label_path) as f:
	label = json.loads(f.read())
print(label['name'])



