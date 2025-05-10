# coding=utf-8
import cv2

from utils import load_cifar10_data, extract_DenseSift_descriptors, build_codebook, input_vector_encoder
from classifier import svm_classifier
from tqdm import  tqdm
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
    # 1. 加载数据集（添加进度条和print指示）
    print("开始加载数据集")

    # 先获取训练集和测试集的长度
    with open('../cifar10/train/train.txt', 'r') as f:
        train_paths = f.readlines()
    with open('../cifar10/test/test.txt', 'r') as f:
        test_paths = f.readlines()
    total_images = len(train_paths) + len(test_paths)
    print(f"总图像数: {total_images} (训练集: {len(train_paths)}, 测试集: {len(test_paths)})")

    # 使用 tqdm 显示总进度条
    with tqdm(total=total_images, desc="加载数据集总进度", position=0) as pbar:
        x_train, y_train = [], []
        # 训练集进度条（position=1）
        for img_path, label in tqdm([p.strip().split(' ') for p in train_paths], desc="训练集", position=1,
                                    leave=False):
            img = cv2.imread(img_path)
            x_train.append(img)
            y_train.append(label)
            pbar.update(1)

        # 测试集进度条（position=1，覆盖训练集进度条）
        x_test, y_test = [], []
        for img_path, label in tqdm([p.strip().split(' ') for p in test_paths], desc="测试集", position=1, leave=False):
            img = cv2.imread(img_path)
            x_test.append(img)
            y_test.append(label)
            pbar.update(1)

    print("正在提取密集SIFT特征...")
    # 合并训练集和测试集的总进度
    total_features = len(x_train) + len(x_test)
    x_train_feature = []
    x_test_feature = []

    # 外层总进度条（position=0）
    with tqdm(total=total_features, desc="密集SIFT总进度", position=0) as pbar_total:
        # 训练集特征提取（内层进度条，position=1）
        for img in tqdm(x_train, desc="训练集", position=1, leave=False):
            kp, des = extract_DenseSift_descriptors(img)
            x_train_feature.append((kp, des))
            pbar_total.update(1)

        # 测试集特征提取（复用内层进度条，position=1）
        for img in tqdm(x_test, desc="测试集", position=1, leave=False):
            kp, des = extract_DenseSift_descriptors(img)
            x_test_feature.append((kp, des))
            pbar_total.update(1)

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
    # 为编码过程添加进度条
    x_train_encoded = []
    with tqdm(total=len(x_train), desc="训练集编码") as pbar:
        for i in range(len(x_train)):
            encoded = spatial_pyramid_matching(x_train[i], x_train_des[i], codebook, level=PYRAMID_LEVEL)
            x_train_encoded.append(encoded)
            pbar.update(1)

    x_test_encoded = []
    with tqdm(total=len(x_test), desc="测试集编码") as pbar:
        for i in range(len(x_test)):
            encoded = spatial_pyramid_matching(x_test[i], x_test_des[i], codebook, level=PYRAMID_LEVEL)
            x_test_encoded.append(encoded)
            pbar.update(1)

    x_train = np.asarray(x_train_encoded)
    x_test = np.asarray(x_test_encoded)

    print("正在使用SVM分类器进行训练和测试，可能需要较长时间...")
    svm_classifier(x_train, y_train, x_test, y_test)
