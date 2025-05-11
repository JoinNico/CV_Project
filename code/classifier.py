# coding=utf-8
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def get_label(path='../cifar10/train/labels.txt'):
    """ Get cifar10 class label"""
    with open(path, 'r') as f:
        names = f.readlines()
    return [n.strip() for n in names]


def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def svm_classifier(x_train, y_train, x_test, y_test):
    # 超参数调优
    print("网格搜索最优超参数...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    clf = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
    clf.fit(x_train, y_train)

    print(f"最佳参数: {clf.best_params_}")
    print(f"最佳交叉验证分数: {clf.best_score_:.3f}")

    # 保存最佳模型
    with open('best_svm_model.pkl', 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

    # 评估
    print("\n分类结果如下:")
    y_pred = clf.predict(x_test)
    class_names = get_label()

    print(classification_report(y_test, y_pred, target_names=class_names))
    plot_confusion_matrix(y_test, y_pred, class_names)