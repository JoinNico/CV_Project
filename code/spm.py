# coding=utf-8
import cv2
import time
from utils import load_cifar10_data, extract_DenseSift_descriptors, build_codebook, input_vector_encoder, \
    load_data_with_progress
from classifier import svm_classifier
from tqdm import tqdm
import numpy as np
import pickle

VOC_SIZE = 100
PYRAMID_LEVEL = 2
DSIFT_STEP_SIZE = 4

# 在文件开头添加测试模式开关
TEST_MODE = True  # 设为False则使用完整数据集
TEST_SAMPLE_SIZE = 1000  # 测试样本量


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
    # 记录开始时间
    start_time = time.time()

    # 1. 加载数据集（添加进度条和print指示）
    (x_train, y_train), (x_test, y_test) = load_data_with_progress(
        test_mode=TEST_MODE,
        test_sample_size=TEST_SAMPLE_SIZE
    )

    print("正在提取密集SIFT特征...")
    # 记录特征提取开始时间
    feature_start_time = time.time()

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

    feature_time = time.time() - feature_start_time
    print(f"\n密集SIFT特征提取耗时: {feature_time:.2f}秒")

    x_train_kp, x_train_des = zip(*x_train_feature)
    x_test_kp, x_test_des = zip(*x_test_feature)

    print("训练集/测试集划分: {:d}/{:d}".format(len(y_train), len(y_test)))
    print("视觉词典大小: {:d}".format(VOC_SIZE))
    print("金字塔层级: {:d}".format(PYRAMID_LEVEL))
    print("正在构建视觉词典，可能需要较长时间...")

    # 记录词典构建开始时间
    codebook_start_time = time.time()
    codebook = build_codebook(x_train_des, VOC_SIZE)
    codebook_time = time.time() - codebook_start_time
    print(f"视觉词典构建耗时: {codebook_time:.2f}秒")

    with open('./spm_lv1_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    print("正在进行空间金字塔匹配编码...")
    # 记录SPM编码开始时间
    spm_start_time = time.time()

    # 合并训练集和测试集的总进度
    total_encoding = len(x_train) + len(x_test)
    x_train_encoded = []
    x_test_encoded = []

    # 外层总进度条（position=0）
    with tqdm(total=total_encoding, desc="空间金字塔编码总进度", position=0) as pbar_total:
        # 训练集编码（内层进度条，position=1）
        for i in tqdm(range(len(x_train)), desc="训练集", position=1, leave=False):
            encoded = spatial_pyramid_matching(x_train[i], x_train_des[i], codebook, level=PYRAMID_LEVEL)
            x_train_encoded.append(encoded)
            pbar_total.update(1)

        # 测试集编码（复用内层进度条，position=1）
        for i in tqdm(range(len(x_test)), desc="测试集", position=1, leave=False):
            encoded = spatial_pyramid_matching(x_test[i], x_test_des[i], codebook, level=PYRAMID_LEVEL)
            x_test_encoded.append(encoded)
            pbar_total.update(1)

    spm_time = time.time() - spm_start_time
    print(f"空间金字塔匹配编码耗时: {spm_time:.2f}秒")

    x_train = np.asarray(x_train_encoded)
    x_test = np.asarray(x_test_encoded)

    print("正在使用SVM分类器进行训练和测试，可能需要较长时间...")
    # 记录分类开始时间
    classifier_start_time = time.time()
    svm_classifier(x_train, y_train, x_test, y_test)
    classifier_time = time.time() - classifier_start_time
    print(f"SVM分类器训练与测试耗时: {classifier_time:.2f}秒")

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")

    # 显示各阶段时间占比
    print("\n各阶段时间占比:")
    print(f"密集SIFT特征提取: {feature_time / total_time * 100:.1f}%")
    print(f"视觉词典构建: {codebook_time / total_time * 100:.1f}%")
    print(f"空间金字塔匹配编码: {spm_time / total_time * 100:.1f}%")
    print(f"SVM分类器训练与测试: {classifier_time / total_time * 100:.1f}%")