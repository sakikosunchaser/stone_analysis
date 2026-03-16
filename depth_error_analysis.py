import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def calculate_relative_error(z_coin, z_rock_array):
    """
    计算忽略深度差异导致的岩石尺寸估算相对误差。

    Args:
        z_coin (float): 硬币的实际深度 (mm)。
        z_rock_array (np.array): 岩石实际深度的数组 (mm)。

    Returns:
        np.array: 对应于z_rock_array中每个深度的相对误差。
    """
    # 防止除以零的情况，尽管实际物理场景中 z_rock 不会为零
    # 但为数值稳定性，当 z_rock_array 接近零时，误差会非常大
    # 这里我们假设 z_rock_array 中的值都大于0
    relative_error = (z_coin / z_rock_array) - 1
    return relative_error

# 参数设置
# 假设硬币的实际直径 Dc = 25.0 mm (代码中默认值)
Dc = 25.0 
z_coin_fixed = 300.0  # 假设硬币固定在300mm (30cm) 处
# 岩石深度范围从100mm到1000mm (10cm到1m)
z_rock_values = np.linspace(100, 1000, 200)

# 计算相对误差
errors = calculate_relative_error(z_coin_fixed, z_rock_values)

# ---- 绘图 ----
plt.figure(figsize=(12, 7))

# 绘制误差曲线
plt.plot(z_rock_values, errors * 100, label=f'硬币深度 zc = {z_coin_fixed:.0f} mm', color='royalblue', linewidth=2)

# 标记零误差线 (当岩石与硬币同深时)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.text(z_rock_values.mean(), 2, '零误差线 (zr = zc)', va='bottom', ha='center', color='black', fontsize=10)

# 标记硬币的实际位置
plt.axvline(z_coin_fixed, color='red', linestyle=':', linewidth=1.5, label=f'硬币实际位置 (zc = {z_coin_fixed:.0f} mm)')

# 在曲线上标记一些具体的深度点及其误差值
points_of_interest_z_rock = [150, 200, z_coin_fixed, 450, 600, 900] #mm
errors_poi = calculate_relative_error(z_coin_fixed, np.array(points_of_interest_z_rock))

for zr_poi, err_poi in zip(points_of_interest_z_rock, errors_poi):
    plt.plot(zr_poi, err_poi * 100, 'o', color='crimson') # 标记点
    # 在标记点旁边显示误差百分比
    plt.text(zr_poi, 
             err_poi * 100 + (5 if err_poi >=0 else -12), # 根据误差正负调整文本位置避免遮挡
             f'{err_poi*100:.1f}%', 
             ha='center', 
             va='bottom', 
             fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))


# 设置图表标题和坐标轴标签
plt.xlabel('岩石的实际深度 zr (mm)', fontsize=12)
plt.ylabel('岩石直径估算的相对误差 εrel (%)', fontsize=12)
plt.title('忽略深度差异对岩石尺寸估算造成的相对误差 (Dc=25mm)', fontsize=14)

# 显示图例
plt.legend(fontsize=10)

# 添加网格线
plt.grid(True, linestyle='-', alpha=0.6)

# 设置坐标轴的显示范围
plt.xlim(z_rock_values.min(), z_rock_values.max())

# 动态调整y轴的显示范围，使其更具可读性
max_abs_error_display = np.max(np.abs(errors[np.isfinite(errors)])) * 100 # 获取最大绝对误差百分比
relevant_error_limit = np.ceil(max_abs_error_display / 50.0) * 50 # 将y轴限制调整到50的倍数，但至少为50%
relevant_error_limit = max(relevant_error_limit, 50)
if relevant_error_limit > 0:
    plt.ylim(-relevant_error_limit, relevant_error_limit)
else:
    plt.ylim(-100, 100) # 默认范围

# 确保布局紧凑，所有元素都能正确显示
plt.tight_layout()

# 显示图表
# plt.savefig("depth_estimation_error_curve.png") # 如果需要保存图像，取消此行注释
plt.show() 