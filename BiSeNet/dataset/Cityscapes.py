import os
import numpy as np
from torch.utils import data
from PIL import Image
import json
from myutils import encode_segmap


class Cityscapes(data.Dataset):

    def __init__(self, path, pseudo_path=None, crop_size=(1024,512), mean=(104.00698793, 116.66876762, 122.67891434), train=True, max_iters=None, ignore_index=255, ssl=None, train_mode=None):
        self.mean = mean
        self.crop_size = crop_size
        self.train = train
        self.set = 'train' if self.train else 'val'
        self.ignore_index = ignore_index
        self.files = []
        self.ssl = ssl
        self.train_mode = train_mode
        self.path=path

        if self.train: 
            self.img_ids = [i_id.strip() for i_id in open(os.path.join(path, 'train.txt'))]
        else:
            self.img_ids = [i_id.strip() for i_id in open(os.path.join(path, 'val.txt'))]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.info = json.load(open(os.path.join(path, 'info.json'), 'r'))
        self.class_mapping = self.info['label2train']

        for name in self.img_ids:

          names=name.split('/')[1].split('_')
          name = names[0]+'_'+names[1]+'_'+names[2]
          image_path = os.path.join (path,'images',name+'_leftImg8bit.png')
          if (self.ssl is None):
            label_path = os.path.join (path,'labels',name+'_gtFine_labelIds.png')
          else:
              label_path=os.path.join(pseudo_path, name+'_gtFine_labelIds.png')
          self.files.append({
                "image": image_path,
                "label": label_path,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        # open image and label file
        image = Image.open(file['image']).convert('RGB')
        label = Image.open(file['label'])
        name = file['name']

        # resize
        if "train" in self.set: 
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        else:
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)

        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # remap the semantic label
        if not self.ssl:
            label = encode_segmap(label, self.class_mapping, self.ignore_index)

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
