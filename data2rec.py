import mxnet as mx
import numpy as np
import cv2
from src.open_alip import tokenize


index = 0
save_record = mx.recordio.MXIndexedRecordIO('datarec.idx', 'datarec.rec', 'w')
img_data = cv2.imread('img/architecture.jpg')
text_raw = 'Nice to meet you!'
text_caption = 'Best wishes!'
label_raw = tokenize(text_raw).flatten().numpy()
label_caption = tokenize(text_caption).flatten().numpy()
labels = np.concatenate([label_raw, label_caption])
header = mx.recordio.IRHeader(flag=0, label=labels, id=1, id2=0)
new_item = mx.recordio.pack_img(header, img_data)
save_record.write_idx(index, new_item)