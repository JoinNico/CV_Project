# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.font_manager as fm
import os


# 检测系统中文字体
def detect_chinese_font():
    """检测系统中可用的中文字体"""
    chinese_fonts = []
    for f in fm.findSystemFonts():
        try:
            if any(name in f for name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi', 'SimSun', 'NSimSun']):
                chinese_fonts.append(f)
        except:
            pass

    if chinese_fonts:
        return chinese_fonts[0]
    return None


# 设置中文字体
def set_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 获取系统所有字体
        all_fonts = [f.name for f in fm.fontManager.ttflist]

        # 优先尝试常见中文字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei',
                         'Source Han Sans CN', 'Arial Unicode MS']

        for font in chinese_fonts:
            if font in all_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用字体: {font}")
                break
        else:
            print("未找到中文字体，将使用默认字体")
    except:
        print("设置中文字体失败")


def visualize_confusion_matrix(y_true, y_pred, class_names=None, title='混淆矩阵',
                               normalize=True, save_path=None, figsize=(10, 8)):
    """
    可视化混淆矩阵

    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        class_names: 类别名称列表
        title: 图表标题
        normalize: 是否归一化
        save_path: 保存路径，如果不提供则显示图像
        figsize: 图像大小
    """
    # 设置中文字体
    set_chinese_font()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # 创建图表
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)

    # 优化布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存混淆矩阵到: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_classification_report(y_true, y_pred, class_names=None, title='分类报告',
                                    save_path=None, figsize=(10, 8)):
    """
    将分类报告可视化为热力图

    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径，如果不提供则显示图像
        figsize: 图像大小
    """
    # 设置中文字体
    set_chinese_font()

    # 获取分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # 提取需要展示的指标
    metrics = ['precision', 'recall', 'f1-score']

    # 创建数据框
    class_data = {}
    for cls in class_names:
        class_data[cls] = [report[cls][metric] for metric in metrics]

    # 转换为numpy数组
    data = np.array(list(class_data.values()))

    # 创建图表
    plt.figure(figsize=figsize)

    # 绘制热力图
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=metrics, yticklabels=class_names)

    plt.title(title, fontsize=16)
    plt.ylabel('类别', fontsize=12)
    plt.xlabel('评估指标', fontsize=12)

    # 优化布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存分类报告到: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_precision_recall_curve(class_names, precision_values, recall_values,
                                title='精确率-召回率曲线', save_path=None, figsize=(12, 8)):
    """
    绘制精确率-召回率曲线

    参数:
        class_names: 类别名称列表
        precision_values: 各类别的精确率值
        recall_values: 各类别的召回率值
        title: 图表标题
        save_path: 保存路径，如果不提供则显示图像
        figsize: 图像大小
    """
    # 设置中文字体
    set_chinese_font()

    plt.figure(figsize=figsize)

    # 设置颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    # 绘制每个类别的点
    for i, cls in enumerate(class_names):
        plt.scatter(recall_values[i], precision_values[i], color=colors[i], s=100, label=cls)

    # 连接点成线
    for i in range(len(class_names)):
        plt.plot([0, recall_values[i]], [precision_values[i], precision_values[i]],
                 '--', color=colors[i], alpha=0.5)
        plt.plot([recall_values[i], recall_values[i]], [0, precision_values[i]],
                 '--', color=colors[i], alpha=0.5)

    plt.title(title, fontsize=16)
    plt.xlabel('召回率', fontsize=12)
    plt.ylabel('精确率', fontsize=12)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))

    # 优化布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存精确率-召回率曲线到: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_model_comparison(model_names, accuracies, title='模型性能比较',
                          save_path=None, figsize=(10, 6)):
    """
    比较不同模型的性能

    参数:
        model_names: 模型名称列表
        accuracies: 各模型的准确率
        title: 图表标题
        save_path: 保存路径，如果不提供则显示图像
        figsize: 图像大小
    """
    # 设置中文字体
    set_chinese_font()

    plt.figure(figsize=figsize)

    # 绘制条形图
    bars = plt.bar(model_names, accuracies, color='skyblue', edgecolor='black')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, max(accuracies) * 1.1)  # 留出标签空间
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 优化布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存模型比较图到: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_time_breakdown(time_data, title='处理时间占比', save_path=None, figsize=(10, 6)):
    """
    绘制时间占比饼图

    参数:
        time_data: 字典，键为阶段名称，值为时间（秒）
        title: 图表标题
        save_path: 保存路径，如果不提供则显示图像
        figsize: 图像大小
    """
    # 设置中文字体
    set_chinese_font()

    plt.figure(figsize=figsize)

    # 数据准备
    labels = list(time_data.keys())
    values = list(time_data.values())
    total_time = sum(values)

    # 计算百分比
    percentages = [v / total_time * 100 for v in values]

    # 自定义颜色
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))

    # 绘制饼图
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        shadow=False,
        explode=[0.05] * len(labels),
        textprops={'fontsize': 12}
    )

    # 调整自动百分比文本
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

    plt.title(title, fontsize=16)
    plt.axis('equal')  # 保持圆形

    # 添加图例显示具体时间
    legend_labels = [f'{l}: {v:.1f}秒 ({p:.1f}%)' for l, v, p in zip(labels, values, percentages)]
    plt.legend(wedges, legend_labels, title="时间明细", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 优化布局
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存时间占比图到: {save_path}")
    else:
        plt.show()
    plt.close()


def create_visualization_directory(base_dir='./visualization'):
    """创建可视化结果保存目录"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"创建目录: {base_dir}")

    # 创建子目录
    subdirs = ['bow', 'spm', 'comparison']
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"创建目录: {path}")

    return base_dir