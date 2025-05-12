# coding=utf-8
import os
import argparse
from comparison import compare_models


def main():
    """主函数：解析命令行参数并运行模型比较"""
    parser = argparse.ArgumentParser(description='BOW和SPM模型的可视化比较')

    # 基本参数
    parser.add_argument('--test_mode', action='store_true', default=True,
                        help='是否使用测试模式（默认：True）')
    parser.add_argument('--test_sample_size', type=int, default=1000,
                        help='测试样本数量（默认：1000）')

    # 模型选择参数
    parser.add_argument('--run_bow', action='store_true', default=True,
                        help='是否运行BOW模型（默认：True）')
    parser.add_argument('--run_spm', action='store_true', default=True,
                        help='是否运行SPM模型（默认：True）')
    parser.add_argument('--spm_levels', type=int, nargs='+', default=[1],
                        help='SPM模型的金字塔层级列表（默认：1 2）')

    # 可视化参数
    parser.add_argument('--visualization_dir', type=str, default='./visualization',
                        help='可视化结果保存目录（默认：./visualization）')

    # 解析参数
    args = parser.parse_args()

    # 创建可视化目录
    if not os.path.exists(args.visualization_dir):
        os.makedirs(args.visualization_dir)

    # 打印参数
    print("\n===== 运行参数 =====")
    print(f"测试模式: {args.test_mode}")
    print(f"测试样本数量: {args.test_sample_size}")
    print(f"运行BOW模型: {args.run_bow}")
    print(f"运行SPM模型: {args.run_spm}")
    print(f"SPM金字塔层级: {args.spm_levels}")
    print(f"可视化结果保存目录: {args.visualization_dir}")
    print("=" * 20)

    # 运行模型比较
    results = compare_models(
        test_mode=args.test_mode,
        test_sample_size=args.test_sample_size,
        run_bow=args.run_bow,
        run_spm=args.run_spm,
        spm_levels=args.spm_levels
    )

    print("\n所有任务已完成！")


if __name__ == '__main__':
    main()