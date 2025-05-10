# coding=utf-8
import pickle
import numpy as np
from tqdm import tqdm
from utils import load_cifar10_data, extract_sift_descriptors, build_codebook, input_vector_encoder
from classifier import svm_classifier

VOC_SIZE = 100

if __name__ == '__main__':
    # 1. 加载数据集（总进度条）
    print("加载数据集...")

    # 先获取训练集和测试集的长度
    with open('../cifar10/train/train.txt', 'r') as f:
        train_paths = f.readlines()
    with open('../cifar10/test/test.txt', 'r') as f:
        test_paths = f.readlines()

    total_images = len(train_paths) + len(test_paths)

    with tqdm(total=total_images, desc="总进度") as pbar:
        # 加载训练集
        with tqdm(total=len(train_paths), desc="加载训练集", leave=False) as pbar_train:
            x_train, y_train = load_cifar10_data(dataset='train', pbar=pbar_train)
            pbar.update(len(x_train))  # 更新总进度条

        # 加载测试集
        with tqdm(total=len(test_paths), desc="加载测试集", leave=False) as pbar_test:
            x_test, y_test = load_cifar10_data(dataset='test', pbar=pbar_test)
            pbar.update(len(x_test))  # 更新总进度条

    # 2. 提取 SIFT 特征（总进度条）
    print("\n提取 SIFT 特征...")
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
    with tqdm(total=1, desc="总进度") as pbar:
        codebook = build_codebook(x_train_des, voc_size=VOC_SIZE)
        pbar.update(1)
    with open('./bow_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    # 5. 特征编码（总进度条）
    print("\n进行特征编码...")
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

    x_train_encoded = np.array(x_train_encoded)
    x_test_encoded = np.array(x_test_encoded)

    # 6. 训练分类器（总进度条）
    print("\n训练分类器...")
    with tqdm(total=1, desc="总进度") as pbar:
        svm_classifier(x_train_encoded, y_train, x_test_encoded, y_test)
        pbar.update(1)