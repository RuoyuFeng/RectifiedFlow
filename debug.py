import numpy as np
import matplotlib.pyplot as plt

# 定义参数
T = 1.0  # 你可以根据需要调整
s = 0.008  # 你可以根据需要调整

# 定义 f(t) 函数
def f(t, T, s):
    return (np.cos(((t/T + s) / (1 + s) * np.pi / 2))) ** 2

# 定义 \bar{\alpha}_t 函数
def alpha_bar_t(t, T, s):
    return f(t, T, s) / f(0, T, s)

# 生成时间点
t_values = np.linspace(0, 1, 500)

# 计算 \bar{\alpha}_t 的值
alpha_bar_values = alpha_bar_t(t_values, T, s)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(t_values, alpha_bar_values, label=r'$\bar{\alpha}_t$')
plt.xlabel('t')
plt.ylabel(r'$\bar{\alpha}_t$')
plt.title(r'Visualization of $\bar{\alpha}_t = \frac{f(t)}{f(0)}$')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('alpha_bar_t.png')
