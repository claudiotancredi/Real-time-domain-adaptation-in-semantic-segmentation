import numpy as np

def encode_segmap(mask, mapping, ignore_index):
    label_copy = ignore_index * np.ones(mask.shape, dtype=np.float32)
    for k, v in mapping:
        label_copy[mask == k] = v

    return label_copy

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = image.astype(int)

    final = np.zeros((x.shape[0], x.shape[1],3), np.ubyte)
    x[x==255]=19
    final[:,:]=colour_codes[x]
    return final

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(optimizer, i_iter, lrate, num_steps, power):
    lr = lr_poly(lrate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10