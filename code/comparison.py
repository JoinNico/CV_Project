# coding=utf-8
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# 导入自定义模块
from bow import run_bow_with_visualization
from spm import run_spm_with_visualization
from visualization import (
    plot_model_comparison,
    visualize_confusion_matrix,
    set_chinese_font,
    create_visualization_directory
)
from classifier import get_label


def compare_models(test_mode=True, test_sample_size=1000, run_bow=True, run_spm=True, spm_levels=[1, 2]):
    """
    比较BOW和SPM模型性能

    参数:
        test_mode: 是否使用测试模式
        test_sample_size: 测试样本数量
        run_bow: 是否运行BOW模型
        run_spm: 是否运行SPM模型
        spm_levels: SPM模型的金字塔层级列表
    """
    # 创建可视化目录
    vis_dir = create_visualization_directory()
    comp_dir = os.path.join(vis_dir, 'comparison')

    results = []

    # 运行BOW模型
    if run_bow:
        print("\n" + "=" * 50)
        print("运行BOW模型...")
        print("=" * 50)
        bow_result = run_bow_with_visualization(
            test_mode=test_mode,
            test_sample_size=test_sample_size
        )
        results.append(bow_result)

    # 运行SPM模型
    if run_spm:
        for level in spm_levels:
            print("\n" + "=" * 50)
            print(f"运行SPM模型 (层级:{level})...")
            print("=" * 50)
            spm_result = run_spm_with_visualization(
                test_mode=test_mode,
                test_sample_size=test_sample_size,
                pyramid_level=level
            )
            results.append(spm_result)

    if not results:
        print("没有运行任何模型，无法进行比较")
        return

    # 比较模型性能
    print("\n" + "=" * 50)
    print("开始比较模型性能...")
    print("=" * 50)

    # 提取模型名称和准确率
    model_names = [r['model'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    # 绘制模型比较图
    comparison_path = os.path.join(comp_dir, 'model_accuracy_comparison.png')
    plot_model_comparison(
        model_names, accuracies,
        title='模型准确率比较',
        save_path=comparison_path
    )

    # 绘制混淆矩阵比较（如果只有两个模型）
    if len(results) == 2:
        set_chinese_font()
        plt.figure(figsize=(15, 7))

        # 获取类别名称
        class_names = get_label()

        # 绘制第一个模型的混淆矩阵
        plt.subplot(1, 2, 1)
        cm1 = confusion_matrix(results[0]['y_true'], results[0]['y_pred'])
        cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm1, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{results[0]['model']} 混淆矩阵", fontsize=12)
        plt.ylabel('真实标签', fontsize=10)
        plt.xlabel('预测标签', fontsize=10)

        # 绘制第二个模型的混淆矩阵
        plt.subplot(1, 2, 2)
        cm2 = confusion_matrix(results[1]['y_true'], results[1]['y_pred'])
        cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm2, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{results[1]['model']} 混淆矩阵", fontsize=12)
        plt.ylabel('真实标签', fontsize=10)
        plt.xlabel('预测标签', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, 'confusion_matrix_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存混淆矩阵比较图到: {os.path.join(comp_dir, 'confusion_matrix_comparison.png')}")

    # 绘制处理时间比较
    set_chinese_font()
    plt.figure(figsize=(12, 8))

    # 准备时间数据
    time_data = []
    for r in results:
        model_time = r['time_metrics']
        model_time['总时间'] = r['total_time']
        time_data.append({
            'model': r['model'],
            **model_time
        })

    # 将时间数据转换为数据框
    df = pd.DataFrame(time_data)
    df_melted = pd.melt(df, id_vars=['model'], var_name='阶段', value_name='时间(秒)')

    # 绘制柱状图
    sns.barplot(x='阶段', y='时间(秒)', hue='model', data=df_melted)
    plt.title('各模型处理时间比较', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='模型')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图像
    time_comparison_path = os.path.join(comp_dir, 'time_comparison.png')
    plt.savefig(time_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存时间比较图到: {time_comparison_path}")

    # 输出比较结果表格
    print("\n模型性能比较:")
    result_table = []
    for r in results:
        result_table.append({
            '模型': r['model'],
            '准确率': f"{r['accuracy']:.4f}",
            '总处理时间(秒)': f"{r['total_time']:.2f}"
        })

    # 使用pandas输出格式化的表格
    print(pd.DataFrame(result_table).to_string(index=False))
    print(f"\n比较结果已保存到目录: {comp_dir}")

    return results


def visualize_class_performance_comparison(results, save_dir):
    """
    可视化不同模型在各个类别上的性能比较

    参数:
        results: 模型结果列表
        save_dir: 保存目录
    """
    class_names = get_label()
    model_names = [r['model'] for r in results]

    # 计算每个模型在每个类别上的准确率
    class_performance = []

    for r in results:
        y_true = np.array(r['y_true'])
        y_pred = np.array(r['y_pred'])

        for i, class_name in enumerate(class_names):
            # 找出属于该类别的样本索引
            class_indices = np.array([j for j, label in enumerate(y_true) if label == class_name])
            if len(class_indices) > 0:
                # 计算该类别的准确率
                class_correct = sum(1 for j in class_indices if y_pred[j] == y_true[j])
                class_accuracy = class_correct / len(class_indices)
            else:
                class_accuracy = 0

            class_performance.append({
                '模型': r['model'],
                '类别': class_name,
                '准确率': class_accuracy
            })

    # 创建数据框
    df = pd.DataFrame(class_performance)

    # 绘制热力图
    set_chinese_font()
    plt.figure(figsize=(12, 8))

    # 透视表
    pivot_df = df.pivot(index='类别', columns='模型', values='准确率')

    # 绘制热力图
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': '准确率'})
    plt.title('各模型在不同类别上的准确率比较', fontsize=14)
    plt.tight_layout()

    # 保存图像
    class_comparison_path = os.path.join(save_dir, 'class_performance_comparison.png')
    plt.savefig(class_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存类别性能比较图到: {class_comparison_path}")

    return df


if __name__ == '__main__':
    # 比较模型性能
    results = compare_models(
        test_mode=True,
        test_sample_size=1000,
        run_bow=True,
        run_spm=True,
        spm_levels=[1, 2]
    )

    # 如果有结果，比较类别性能
    if results:
        vis_dir = create_visualization_directory()
        comp_dir = os.path.join(vis_dir, 'comparison')
        visualize_class_performance_comparison(results, comp_dir)