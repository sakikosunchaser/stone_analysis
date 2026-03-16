import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

def compare_results():
    """对比改进前后的结果"""
    results_dir = "results"
    
    # 找到所有结果图像
    result_files = glob.glob(os.path.join(results_dir, "*_result_*.jpg"))
    if len(result_files) < 2:
        print("需要至少2个结果图像进行对比")
        return
    
    # 按修改时间排序
    result_files.sort(key=os.path.getmtime)
    
    # 选择最新的两个结果
    old_result = result_files[-2]  # 倒数第二个
    new_result = result_files[-1]  # 最新的
    
    # 加载图像
    old_img = Image.open(old_result)
    new_img = Image.open(new_result)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示旧结果
    axes[0].imshow(old_img)
    axes[0].set_title(f'改进前: {os.path.basename(old_result)}', fontsize=12)
    axes[0].axis('off')
    
    # 显示新结果
    axes[1].imshow(new_img)
    axes[1].set_title(f'改进后: {os.path.basename(new_result)}', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('可视化效果对比', fontsize=16, y=0.98)
    plt.show()
    
    print(f"对比图像:")
    print(f"  改进前: {old_result}")
    print(f"  改进后: {new_result}")

if __name__ == "__main__":
    compare_results() 