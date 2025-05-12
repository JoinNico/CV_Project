# coding=utf-8
import os
import numpy as np
import time
import cv2
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 导入自定义模块
from utils import load_cifar10_data, extract_DenseSift_descriptors, build_codebook, load_data_with_progress, \
    DSIFT_STEP_SIZE, input_vector_encoder
from classifier import svm_classifier, get_label
from visualization import (
    visualize_confusion_matrix,
    visualize_classification_report,
    plot_precision_recall_curve,
    plot_time_breakdown,
    create_visualization_directory
)

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

def run_spm_with_visualization(test_mode=True, test_sample_size=1000, voc_size=100, pyramid_level=2):
    """
    运行Spatial Pyramid Matching模型并生成可视化结果

    参数:
        test_mode: 是否使用测试模式
        test_sample_size: 测试样本数量
        voc_size: 视觉词典大小
        pyramid_level: 金字塔层级
    """
    # 创建可视化目录
    vis_dir = create_visualization_directory()
    spm_dir = os.path.join(vis_dir, 'spm')

    # 记录时间
    start_time = time.time()
    time_metrics = {}

    # 1. 加载数据集
    print("\n开始加载数据...")
    data_start_time = time.time()
    (x_train, y_train), (x_test, y_test) = load_data_with_progress(
        test_mode=test_mode,
        test_sample_size=test_sample_size
    )
    data_time = time.time() - data_start_time
    time_metrics['数据加载'] = data_time
    print(f"数据加载完成，耗时: {data_time:.2f}秒")

    # 2. 提取密集SIFT特征
    print("\n开始提取密集SIFT特征...")
    feature_start_time = time.time()

    # 合并训练集和测试集的总进度
    total_features = len(x_train) + len(x_test)
    x_train_feature = []
    x_test_feature = []

    # 外层总进度条
    with tqdm(total=total_features, desc="密集SIFT总进度") as pbar_total:
        # 训练集特征提取
        for img in tqdm(x_train, desc="训练集", leave=False):
            kp, des = extract_DenseSift_descriptors(img)
            x_train_feature.append((kp, des))
            pbar_total.update(1)

        # 测试集特征提取
        for img in tqdm(x_test, desc="测试集", leave=False):
            kp, des = extract_DenseSift_descriptors(img)
            x_test_feature.append((kp, des))
            pbar_total.update(1)

    feature_time = time.time() - feature_start_time
    time_metrics['密集SIFT特征提取'] = feature_time
    print(f"密集SIFT特征提取完成，耗时: {feature_time:.2f}秒")

    x_train_kp, x_train_des = zip(*x_train_feature)
    x_test_kp, x_test_des = zip(*x_test_feature)

    print(f"训练集/测试集划分: {len(y_train)}/{len(y_test)}")
    print(f"视觉词典大小: {voc_size}")
    print(f"金字塔层级: {pyramid_level}")

    # 3. a构建视觉词典
    print("\n开始构建视觉词典...")
    codebook_start_time = time.time()
    codebook = build_codebook(x_train_des, voc_size)
    codebook_time = time.time() - codebook_start_time
    time_metrics['视觉词典构建'] = codebook_time
    print(f"视觉词典构建完成，耗时: {codebook_time:.2f}秒")

    # 保存词典
    with open('./spm_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    # 4. 进行空间金字塔匹配编码
    print("\n开始进行空间金字塔匹配编码...")
    spm_start_time = time.time()

    # 合并训练集和测试集的总进度
    total_encoding = len(x_train) + len(x_test)
    x_train_encoded = []
    x_test_encoded = []

    # 外层总进度条
    with tqdm(total=total_encoding, desc="空间金字塔编码总进度") as pbar_total:
        # 训练集编码
        for i in tqdm(range(len(x_train)), desc="训练集", leave=False):
            encoded = spatial_pyramid_matching(x_train[i], x_train_des[i], codebook, level=pyramid_level)
            x_train_encoded.append(encoded)
            pbar_total.update(1)

        # 测试集编码
        for i in tqdm(range(len(x_test)), desc="测试集", leave=False):
            encoded = spatial_pyramid_matching(x_test[i], x_test_des[i], codebook, level=pyramid_level)
            x_test_encoded.append(encoded)
            pbar_total.update(1)

    spm_time = time.time() - spm_start_time
    time_metrics['空间金字塔匹配编码'] = spm_time
    print(f"空间金字塔匹配编码完成，耗时: {spm_time:.2f}秒")

    x_train_encoded = np.asarray(x_train_encoded)
    x_test_encoded = np.asarray(x_test_encoded)

    # 5. 训练分类器并预测
    print("\n开始训练分类器...")
    classifier_start_time = time.time()

    # 获取预测结果
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(x_train_encoded, y_train)
    y_pred = clf.predict(x_test_encoded)

    classifier_time = time.time() - classifier_start_time
    time_metrics['SVM分类器训练与测试'] = classifier_time
    print(f"SVM分类器训练与测试完成，耗时: {classifier_time:.2f}秒")

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")

    # 6. 可视化结果
    print("\n开始生成可视化结果...")
    class_names = get_label()

    # 6.1 混淆矩阵
    confusion_matrix_path = os.path.join(spm_dir, 'confusion_matrix.png')
    visualize_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        title=f'SPM模型混淆矩阵 (层级:{pyramid_level})',
        save_path=confusion_matrix_path
    )

    # 6.2 分类报告可视化
    report_path = os.path.join(spm_dir, 'classification_report.png')
    visualize_classification_report(
        y_test, y_pred,
        class_names=class_names,
        title=f'SPM模型分类报告 (层级:{pyramid_level})',
        save_path=report_path
    )

    # 6.3 精确率-召回率曲线
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)
    pr_curve_path = os.path.join(spm_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(
        class_names, precision, recall,
        title=f'SPM模型各类别精确率-召回率 (层级:{pyramid_level})',
        save_path=pr_curve_path
    )

    # 6.4 时间占比饼图
    time_path = os.path.join(spm_dir, 'time_breakdown.png')
    plot_time_breakdown(
        time_metrics,
        title=f'SPM模型处理时间占比 (层级:{pyramid_level})',
        save_path=time_path
    )

    # 7. 返回结果摘要
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSPM模型准确率: {accuracy:.4f}")
    print(f"可视化结果已保存到目录: {spm_dir}")

    return {
        'model': f'SPM (层级:{pyramid_level})',
        'accuracy': accuracy,
        'y_true': y_test,
        'y_pred': y_pred,
        'time_metrics': time_metrics,
        'total_time': total_time
    }


if __name__ == '__main__':
    # 运行SPM模型并可视化
    run_spm_with_visualization(test_mode=True, test_sample_size=1000, pyramid_level=2)



