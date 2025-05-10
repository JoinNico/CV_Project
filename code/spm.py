# coding=utf-8
from utils import load_cifar10_data, extract_DenseSift_descriptors, build_codebook, input_vector_encoder
from classifier import svm_classifier

import numpy as np
import pickle

VOC_SIZE = 100
PYRAMID_LEVEL = 2
DSIFT_STEP_SIZE = 4


def build_spatial_pyramid(image, descriptor, level):
    """根据金字塔层级重建描述符"""
    assert 0 <= level <= 2, "金字塔层级参数错误"
    step_size = DSIFT_STEP_SIZE
    from utils import DSIFT_STEP_SIZE as s
    assert s == step_size, "步长必须与utils.extract_DenseSift_descriptors()中的DSIFT_STEP_SIZE一致"

    h = image.shape[0] // step_size
    w = image.shape[1] // step_size

    idx_crop = np.array(range(len(descriptor))).reshape(h, w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2 ** (3 - level), 2 ** (3 - level)
    shape = (height // bh, width // bw, bh, bw)
    strides = size * np.array([width * bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
        idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid


def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))


if __name__ == '__main__':
    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    print("正在提取密集SIFT特征...")
    x_train_feature = [extract_DenseSift_descriptors(img) for img in x_train]
    x_test_feature = [extract_DenseSift_descriptors(img) for img in x_test]
    x_train_kp, x_train_des = zip(*x_train_feature)
    x_test_kp, x_test_des = zip(*x_test_feature)

    print("训练集/测试集划分: {:d}/{:d}".format(len(y_train), len(y_test)))
    print("视觉词典大小: {:d}".format(VOC_SIZE))
    print("金字塔层级: {:d}".format(PYRAMID_LEVEL))
    print("正在构建视觉词典，可能需要较长时间...")
    codebook = build_codebook(x_train_des, VOC_SIZE)

    with open('./spm_lv1_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    print("正在进行空间金字塔匹配编码...")
    x_train = [spatial_pyramid_matching(x_train[i],
                                        x_train_des[i],
                                        codebook,
                                        level=PYRAMID_LEVEL)
               for i in range(len(x_train))]

    x_test = [spatial_pyramid_matching(x_test[i],
                                       x_test_des[i],
                                       codebook,
                                       level=PYRAMID_LEVEL)
              for i in range(len(x_test))]

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    print("正在使用SVM分类器进行训练和测试，可能需要较长时间...")
    svm_classifier(x_train, y_train, x_test, y_test)
