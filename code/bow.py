# coding=utf-8
import pickle
import time
import cv2
import numpy as np
from tqdm import tqdm
from utils import load_cifar10_data, extract_sift_descriptors, build_codebook, input_vector_encoder, \
    load_data_with_progress
from classifier import svm_classifier

VOC_SIZE = 100

# 在文件开头添加测试模式开关
TEST_MODE = True  # 设为False则使用完整数据集
TEST_SAMPLE_SIZE = 1000  # 测试样本量

if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()

    # 1. 加载数据集（总进度条）
    (x_train, y_train), (x_test, y_test) = load_data_with_progress(
        test_mode=TEST_MODE,
        test_sample_size=TEST_SAMPLE_SIZE
    )

    # 2. 提取 SIFT 特征（总进度条）
    print("\n提取 SIFT 特征...")
    feature_start_time = time.time()
    with tqdm(total=len(x_train) + len(x_test), desc="总进度") as pbar:
        x_train_des = []
        for img in tqdm(x_train, desc="训练集特征提取", leave=False):
            des = extract_sift_descriptors(img)
            x_train_des.append(des)
            pbar.update(1)

        x_test_des = []
        for img in tqdm(x_test, desc="测试集特征提取", leave=False):
            des = extract_sift_descriptors(img)
            x_test_des.append(des)
            pbar.update(1)
    feature_time = time.time() - feature_start_time
    print(f"\n特征提取耗时: {feature_time:.2f}秒")

    # 3. 过滤无效数据
    train_data = [(des, label) for des, label in zip(x_train_des, y_train) if des is not None and des.size > 0]
    test_data = [(des, label) for des, label in zip(x_test_des, y_test) if des is not None and des.size > 0]

    if not train_data or not test_data:
        raise ValueError("过滤后无有效特征描述符")

    x_train_des, y_train = zip(*train_data)
    x_test_des, y_test = zip(*test_data)
    print(f"\n训练样本数: {len(y_train)}, 测试样本数: {len(y_test)}")

    # 4. 构建视觉词典（总进度条）
    print("\n构建视觉词典...")
    codebook_start_time = time.time()
    codebook = build_codebook(x_train_des, voc_size=VOC_SIZE)
    codebook_time = time.time() - codebook_start_time
    print(f"词典构建耗时: {codebook_time:.2f}秒")

    with open('./bow_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    # 5. 特征编码（总进度条）
    print("\n进行特征编码...")
    encoding_start_time = time.time()
    with tqdm(total=len(x_train_des) + len(x_test_des), desc="总进度") as pbar:
        x_train_encoded = []
        for des in tqdm(x_train_des, desc="训练集编码", leave=False):
            encoded = input_vector_encoder(des, codebook)
            x_train_encoded.append(encoded)
            pbar.update(1)

        x_test_encoded = []
        for des in tqdm(x_test_des, desc="测试集编码", leave=False):
            encoded = input_vector_encoder(des, codebook)
            x_test_encoded.append(encoded)
            pbar.update(1)
    encoding_time = time.time() - encoding_start_time
    print(f"特征编码耗时: {encoding_time:.2f}秒")

    x_train_encoded = np.array(x_train_encoded)
    x_test_encoded = np.array(x_test_encoded)

    # 6. 训练分类器（总进度条）
    print("\n训练分类器...")
    classifier_start_time = time.time()
    svm_classifier(x_train_encoded, y_train, x_test_encoded, y_test)
    classifier_time = time.time() - classifier_start_time
    print(f"分类器训练与测试耗时: {classifier_time:.2f}秒")

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")

    # 显示各阶段时间占比
    print("\n各阶段时间占比:")
    print(f"特征提取: {feature_time / total_time * 100:.1f}%")
    print(f"词典构建: {codebook_time / total_time * 100:.1f}%")
    print(f"特征编码: {encoding_time / total_time * 100:.1f}%")
    print(f"分类训练与测试: {classifier_time / total_time * 100:.1f}%")