# coding=utf-8
import cv2
import numpy as np
from sklearn.cluster import KMeans
import scipy.cluster.vq as vq
from pathlib import Path
import joblib
import os

# 密集SIFT特征的步长（像素间隔）
DSIFT_STEP_SIZE = 4
CACHE_DIR = "./cache"


def ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_cifar10_data(dataset):
    """
    加载CIFAR-10数据集
    输入:
        dataset (str): 'train'或'test'，指定加载训练集或测试集
    输出:
        list: [图像列表, 标签列表]
    """
    if dataset not in ['train', 'test']:
        raise ValueError("dataset参数必须是'train'或'test'")

    path_file = f'../cifar10/{dataset}/{dataset}.txt'
    if not Path(path_file).exists():
        raise FileNotFoundError(f"找不到数据集文件: {path_file}")

    with open(path_file, 'r') as f:
        paths = f.readlines()

    x, y = [], []
    for each in paths:
        each = each.strip()
        try:
            path, label = each.split(' ')
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            x.append(img)
            y.append(label)
        except Exception as e:
            print(f"处理数据行'{each}'时出错: {str(e)}")
            continue

    if not x:
        raise ValueError("加载的数据集为空")
    return [x, y]


def extract_sift_descriptors(img, use_dense=False):
    """
    提取图像的SIFT特征描述符
    输入:
        img (numpy array): BGR格式的图像数组 (H,W,3)
        use_dense (bool): 是否使用密集SIFT
    输出:
        numpy array: 特征描述符数组 (N,128)
    """
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须是BGR格式的3通道numpy数组")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    if use_dense:
        # 密集SIFT
        keypoints = [cv2.KeyPoint(x, y, DSIFT_STEP_SIZE)
                     for y in range(0, gray.shape[0], DSIFT_STEP_SIZE)
                     for x in range(0, gray.shape[1], DSIFT_STEP_SIZE)]
        _, descriptors = sift.compute(gray, keypoints)
    else:
        # 常规SIFT
        _, descriptors = sift.detectAndCompute(gray, None)

    return descriptors if descriptors is not None else np.zeros((0, 128), dtype=np.float32)


def build_codebook(X, voc_size, use_cache=True):
    """
    使用K-means构建视觉词典（码本）
    """
    ensure_cache_dir()
    cache_file = f"{CACHE_DIR}/codebook_voc{voc_size}.pkl"

    if use_cache and os.path.exists(cache_file):
        print(f"从缓存加载码本: {cache_file}")
        return joblib.load(cache_file)

    valid_descs = [desc for desc in X
                   if isinstance(desc, np.ndarray) and desc.ndim == 2 and desc.shape[1] == 128]

    if not valid_descs:
        raise ValueError("未找到可用于构建词典的有效特征描述符")

    features = np.vstack(valid_descs)
    kmeans = KMeans(n_clusters=voc_size, n_init=10, random_state=42)
    kmeans.fit(features)

    if use_cache:
        joblib.dump(kmeans.cluster_centers_, cache_file)

    return kmeans.cluster_centers_


def input_vector_encoder(feature, codebook):
    """
    将局部特征编码为视觉单词直方图
    """
    if feature.size == 0:
        return np.zeros(codebook.shape[0])

    code, _ = vq.vq(feature, codebook)
    word_hist, _ = np.histogram(code, bins=range(codebook.shape[0] + 1), density=True)
    return word_hist