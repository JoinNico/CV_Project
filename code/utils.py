# coding=utf-8
import cv2
import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.vq as vq
import numpy as np
from tqdm import tqdm

# 密集SIFT特征的步长（像素间隔）
DSIFT_STEP_SIZE = 4


def load_cifar10_data(dataset, pbar=None):
    """
    加载CIFAR-10数据集
    输入:
        dataset (str): 'train'或'test'，指定加载训练集或测试集
        pbar (tqdm object): 进度条对象，用于更新加载进度
    输出:
        list: [图像列表, 标签列表]
        图像列表: list of numpy arrays (H,W,3)
        标签列表: list of str
    """
    if dataset == 'train':
        with open('../cifar10/train/train.txt', 'r') as f:
            paths = f.readlines()
    elif dataset == 'test':
        with open('../cifar10/test/test.txt', 'r') as f:
            paths = f.readlines()
    else:
        raise ValueError("dataset must be 'train' or 'test'")

    x, y = [], []
    for each in paths:
        each = each.strip()
        path, label = each.split(' ')
        img = cv2.imread(path)  # 读取BGR格式图像
        x.append(img)
        y.append(label)
        if pbar:  # 如果提供了进度条对象，则更新进度
            pbar.update(1)

    return [x, y]


def extract_sift_descriptors(img):
    """
    提取图像的SIFT特征描述符
    输入:
        img (numpy array): BGR格式的图像数组 (H,W,3)
    输出:
        numpy array: 特征描述符数组 (N,128)，每行是一个128维的SIFT特征向量
                     若无特征则返回空数组 (0,128)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors if descriptors is not None else np.zeros((0, 128), dtype=np.float32)


def extract_DenseSift_descriptors(img):
    """
    提取图像的密集SIFT特征（在固定网格位置提取）
    输入:
        img (numpy array): BGR格式的图像数组 (H,W,3)
    输出:
        tuple: (关键点列表, 描述符数组)
            关键点列表: list of cv2.KeyPoint
            描述符数组: numpy array (N,128)，若无特征则返回 (0,128)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # 在固定网格位置创建关键点
    keypoints = [cv2.KeyPoint(x, y, DSIFT_STEP_SIZE)
                 for y in range(0, gray.shape[0], DSIFT_STEP_SIZE)
                 for x in range(0, gray.shape[1], DSIFT_STEP_SIZE)]
    _, descriptors = sift.compute(gray, keypoints)
    return (keypoints, descriptors if descriptors is not None else np.zeros((0, 128), dtype=np.float32))


def load_data_with_progress(test_mode=False, test_sample_size=10):
    """
    带进度条的数据加载函数
    参数:
        test_mode: 是否测试模式
        test_sample_size: 测试样本量
    返回:
        (x_train, y_train), (x_test, y_test)
    """
    print("加载数据集...")

    # 获取文件路径
    with open('../cifar10/train/train.txt', 'r') as f:
        train_paths = f.readlines()
    with open('../cifar10/test/test.txt', 'r') as f:
        test_paths = f.readlines()

    # 测试模式下减少数据量
    if test_mode:
        train_paths = train_paths[:test_sample_size]
        test_paths = test_paths[:test_sample_size // 2]

    total_images = len(train_paths) + len(test_paths)
    print(f"总图像数: {total_images} (训练集: {len(train_paths)}, 测试集: {len(test_paths)})")

    # 使用tqdm加载数据
    x_train, y_train, x_test, y_test = [], [], [], []
    with tqdm(total=total_images, desc="加载进度") as pbar:
        # 加载训练集
        for img_path, label in tqdm([p.strip().split(' ') for p in train_paths],
                                    desc="训练集", leave=False):
            img = cv2.imread(img_path)
            if img is not None:
                x_train.append(img)
                y_train.append(label)
            pbar.update(1)

        # 加载测试集
        for img_path, label in tqdm([p.strip().split(' ') for p in test_paths],
                                    desc="测试集", leave=False):
            img = cv2.imread(img_path)
            if img is not None:
                x_test.append(img)
                y_test.append(label)
            pbar.update(1)

    return (x_train, y_train), (x_test, y_test)

def build_codebook(X, voc_size):
    """
    使用K-means构建视觉词典（码本）
    输入:
        X (list): 特征描述符列表，每个元素是 (N_i,128) 的numpy数组
        voc_size (int): 视觉词典大小（K-means的聚类中心数）
    输出:
        numpy array: 视觉词典/码本 (voc_size,128)
    异常:
        ValueError: 当输入无有效描述符时抛出
    """
    # 过滤无效描述子并确保二维结构
    valid_descs = [desc for desc in X
                   if isinstance(desc, np.ndarray) and desc.ndim == 2 and desc.shape[1] == 128]

    if not valid_descs:
        raise ValueError("未找到可用于构建词典的有效特征描述符")

    features = np.vstack(valid_descs)  # 将所有描述符堆叠成(N,128)
    kmeans = KMeans(n_clusters=voc_size)  # 创建K-means模型
    kmeans.fit(features)
    return kmeans.cluster_centers_  # 返回聚类中心作为视觉单词


def input_vector_encoder(feature, codebook):
    """
    将局部特征编码为视觉单词直方图
    输入:
        feature (numpy array): 单张图像的特征描述符 (N,128)
        codebook (numpy array): 视觉词典 (voc_size,128)
    输出:
        numpy array: 归一化的视觉单词直方图 (voc_size,)
    """
    # 将每个特征映射到最近的视觉单词
    code, _ = vq.vq(feature, codebook)
    # 计算视觉单词出现频率（归一化直方图）
    word_hist, _ = np.histogram(code, bins=range(codebook.shape[0] + 1), density=True)
    return word_hist