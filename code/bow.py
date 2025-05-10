# coding=utf-8
import pickle
from utils import load_cifar10_data, extract_sift_descriptors, build_codebook, input_vector_encoder
from classifier import svm_classifier
import numpy as np

VOC_SIZE = 100

if __name__ == '__main__':
    # 加载数据
    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    # 提取特征并验证
    print("正在提取 SIFT 特征...")
    x_train_des = [extract_sift_descriptors(img) for img in x_train]
    x_test_des = [extract_sift_descriptors(img) for img in x_test]

    # 移除无效数据（已通过utils.py保证不会返回None）
    train_data = [(des, label) for des, label in zip(x_train_des, y_train) if des.size > 0]
    test_data = [(des, label) for des, label in zip(x_test_des, y_test) if des.size > 0]

    if not train_data or not test_data:
        raise ValueError("过滤后无有效特征描述符")

    x_train_des, y_train = zip(*train_data)
    x_test_des, y_test = zip(*test_data)

    print(f"训练样本数: {len(y_train)}, 测试样本数: {len(y_test)}")
    print(f"特征描述符示例形状: {x_train_des[0].shape}")

    # 构建词典
    print("正在构建视觉词典...")
    codebook = build_codebook(x_train_des, voc_size=VOC_SIZE)
    with open('./bow_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    # 特征编码
    print("正在进行特征编码...")
    x_train_encoded = np.array([input_vector_encoder(des, codebook) for des in x_train_des])
    x_test_encoded = np.array([input_vector_encoder(des, codebook) for des in x_test_des])

    # 分类
    print("正在训练分类器...")
    svm_classifier(x_train_encoded, y_train, x_test_encoded, y_test)