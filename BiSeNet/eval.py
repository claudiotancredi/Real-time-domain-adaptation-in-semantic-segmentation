import torch
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from PIL import Image
import json

from dataset.Cityscapes import Cityscapes
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from model.build_BiSeNet import BiSeNet
from myutils import colour_code_segmentation

def eval(model,dataloader, args):
    label_info = json.load(open(os.path.join(args.path_dataset,'info.json'), 'r'))
    label_colors=label_info['palette']
    classes=label_info['label']

    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label, _, name) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            img = colour_code_segmentation(predict, label_colors)
            img = Image.fromarray(img, 'RGB')
            img.save(os.path.join(args.predicted_labels_folder,name[0]+".png"))

            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        tq.close()
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        for i in range(len(classes)):
            print("IoU for class %s: %.3f" % (classes[i], miou_list[i]))
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the folder with pretrained weights of model')
    parser.add_argument('--pth_file', type=str, default=None, required=True, help='The file with the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--path_dataset', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--predicted_labels_folder', type=str, default="", help="Path to folder where to store predicted labels")
    args = parser.parse_args(params)

    # create dataset and dataloader here
    eval_dataset = Cityscapes (path=args.path_dataset, train=False)
    dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)
   
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)

    print('load model from %s ...' % (os.path.join(args.checkpoint_path, args.pth_file)))
    checkpoint = torch.load((os.path.join(args.checkpoint_path, args.pth_file)))
    # load pretrained model if exists
    if 'best' in args.pth_file:
        print ("The best epoch is {}". format(checkpoint['epoch']))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    print('Done!')

    eval(model, dataloader, args)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', '/content/drive/MyDrive/checkpoints_101_sgd', #change path according to folder
        #BEST CASE
        '--pth_file', 'best_crossentropy_loss.pth', 
        #LATEST CASE
        #'--pth_file', 'latest_crossentropy_loss.pth',
        '--path_dataset', '/content/drive/MyDrive/Datasets/Cityscapes',
        '--cuda', '0',
        '--context_path', 'resnet101',
        '--num_classes', '19',
        '--batch_size','1',
        '--predicted_labels_folder', '/content/drive/MyDrive/predicted_labels_101',
    ]
    main(params)
