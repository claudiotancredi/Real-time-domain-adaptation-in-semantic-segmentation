import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from FDA import FDA_source_to_target_np, toimage
import skimage.color as color


class GTA5(data.Dataset):
    def __init__(self, root, transformation=None, target_folder=None, max_iters=None, crop_size=(1024, 512), mean=(104.00698793, 116.66876762, 122.67891434), scale=True, ignore_label=255):
        self.root = root
        self.list_path = osp.join(self.root, "train.txt")
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.transformation=transformation
        self.target_folder=target_folder
        
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        
        #GET TARGET IF TRANSFORMATION ON SOURCE IS APPLIED
        if (transformation is not None):
            self.img_ids_target = [i_id.strip() for i_id in open(osp.join(self.target_folder, "train.txt"))]
            if max_iters is not None:
                self.img_ids_target = self.img_ids_target * int(np.ceil(float(max_iters) / len(self.img_ids_target)))

            self.filestarget = []
            for name in self.img_ids_target:

                names=name.split('/')[1].split('_')
                name = names[0]+'_'+names[1]+'_'+names[2]
                image_path = osp.join (self.target_folder,'images',name+'_leftImg8bit.png')
                self.filestarget.append({
                    "image": image_path,
                    "name": name
                })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        #TRANSFORMATION
        if (self.transformation is not None):
            targetfiles=self.filestarget[index]
            targetimage=Image.open(targetfiles["image"]).convert('RGB')
            targetimage=targetimage.resize(self.crop_size,Image.BICUBIC)
            if (self.transformation=="FDA"):
                targetimage=np.asarray(targetimage, np.float32)
                image=image.transpose((2,0,1))
                targetimage=targetimage.transpose((2,0,1))
                src_in_trg = FDA_source_to_target_np(image, targetimage, L=0.01)
                image=src_in_trg.transpose((1,2,0))
                image=toimage(image, cmin=0.0, cmax=255.0)
                image.save("/content/drive/MyDrive/FDA_source_images/"+name+".png") #can be deleted after one epoch, it's just to save transformed images
            else:
                image = np.asarray(image)/255 #/255 is needed to adjust the range
                image_lab=color.rgb2lab(image)
                mean_s=np.mean(image_lab, axis=(0,1))
                std_s=np.std(image_lab, axis=(0,1))

                img_trg_lab=color.rgb2lab(targetimage)
                mean_t=np.mean(img_trg_lab, axis=(0,1))
                std_t=np.std(img_trg_lab, axis=(0,1))

                image_lab_transformed=((image_lab-mean_s)/std_s)*std_t+mean_t
                image=color.lab2rgb(image_lab_transformed)*255 #*255 is needed to adjust the range
                image = image.astype(np.uint8)
                im = Image.fromarray(image, "RGB") #can be deleted after one epoch, it's just to save transformed images
                im.save("/content/drive/MyDrive/LAB_source_images/"+name+".png") #can be deleted after one epoch, it's just to save transformed images
            image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name
