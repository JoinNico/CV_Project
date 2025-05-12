# coding=utf-8
import os
import numpy as np
import time
import cv2
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 导入自定义模块
from utils import load_cifar10_data, extract_sift_descriptors, build_codebook, input_vector_encoder, \
    load_data_with_progress
from classifier import svm_classifier, get_label
from visualization import (
    visualize_confusion_matrix,
    visualize_classification_report,
    plot_precision_recall_curve,
    plot_time_breakdown,
    create_visualization_directory
)


def run_bow_with_visualization(test_mode=True, test_sample_size=1000, voc_size=100):
    """
    运行Bag of Words模型并生成可视化结果

    参数:
        test_mode: 是否使用测试模式
        test_sample_size: 测试样本数量
        voc_size: 视觉词典大小
    """
    # 创建可视化目录
    vis_dir = create_visualization_directory()
    bow_dir = os.path.join(vis_dir, 'bow')

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

    # 2. 提取 SIFT 特征
    print("\n开始提取SIFT特征...")
    feature_start_time = time.time()
    with tqdm(total=len(x_train) + len(x_test), desc="特征提取总进度") as pbar:
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
    time_metrics['特征提取'] = feature_time
    print(f"特征提取完成，耗时: {feature_time:.2f}秒")

    # 3. 过滤无效数据
    train_data = [(des, label) for des, label in zip(x_train_des, y_train) if des is not None and des.size > 0]
    test_data = [(des, label) for des, label in zip(x_test_des, y_test) if des is not None and des.size > 0]

    if not train_data or not test_data:
        raise ValueError("过滤后无有效特征描述符")

    x_train_des, y_train = zip(*train_data)
    x_test_des, y_test = zip(*test_data)
    print(f"有效训练样本数: {len(y_train)}, 有效测试样本数: {len(y_test)}")

    # 4. 构建视觉词典
    print("\n开始构建视觉词典...")
    codebook_start_time = time.time()
    codebook = build_codebook(x_train_des, voc_size=voc_size)
    codebook_time = time.time() - codebook_start_time
    time_metrics['视觉词典构建'] = codebook_time
    print(f"视觉词典构建完成，耗时: {codebook_time:.2f}秒")

    # 保存视觉词典
    with open('./bow_codebook.pkl', 'wb') as f:
        pickle.dump(codebook, f)

    # 5. 特征编码
    print("\n开始特征编码...")
    encoding_start_time = time.time()
    with tqdm(total=len(x_train_des) + len(x_test_des), desc="特征编码总进度") as pbar:
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
    time_metrics['特征编码'] = encoding_time
    print(f"特征编码完成，耗时: {encoding_time:.2f}秒")

    x_train_encoded = np.array(x_train_encoded)
    x_test_encoded = np.array(x_test_encoded)

    # 6. 训练分类器并预测
    print("\n开始训练分类器...")
    classifier_start_time = time.time()

    # 获取预测结果
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(x_train_encoded, y_train)
    y_pred = clf.predict(x_test_encoded)

    classifier_time = time.time() - classifier_start_time
    time_metrics['分类器训练与测试'] = classifier_time
    print(f"分类器训练与测试完成，耗时: {classifier_time:.2f}秒")

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")

    # 7. 可视化结果
    print("\n开始生成可视化结果...")
    class_names = get_label()

    # 7.1 混淆矩阵
    confusion_matrix_path = os.path.join(bow_dir, 'confusion_matrix.png')
    visualize_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        title='BOW模型混淆矩阵',
        save_path=confusion_matrix_path
    )

    # 7.2 分类报告可视化
    report_path = os.path.join(bow_dir, 'classification_report.png')
    visualize_classification_report(
        y_test, y_pred,
        class_names=class_names,
        title='BOW模型分类报告',
        save_path=report_path
    )

    # 7.3 精确率-召回率曲线
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)
    pr_curve_path = os.path.join(bow_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(
        class_names, precision, recall,
        title='BOW模型各类别精确率-召回率',
        save_path=pr_curve_path
    )

    # 7.4 时间占比饼图
    time_path = os.path.join(bow_dir, 'time_breakdown.png')
    plot_time_breakdown(
        time_metrics,
        title='BOW模型处理时间占比',
        save_path=time_path
    )

    # 8. 返回结果摘要
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBOW模型准确率: {accuracy:.4f}")
    print(f"可视化结果已保存到目录: {bow_dir}")

    return {
        'model': 'BOW',
        'accuracy': accuracy,
        'y_true': y_test,
        'y_pred': y_pred,
        'time_metrics': time_metrics,
        'total_time': total_time
    }


if __name__ == '__main__':
    # 运行BOW模型并可视化
    run_bow_with_visualization(test_mode=True, test_sample_size=1000)