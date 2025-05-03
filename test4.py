import numpy as np

def polyfit_no_const(x, y, n):
    """
    最小二乘拟合 y ≈ Σ_{k=1..n} a_k x^k，
    常数项固定为 0，返回系数数组 a[1], …, a[n]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # ① 生成 n 列：x¹, x², …, xⁿ
    # 等价于：A = np.column_stack([x**k for k in range(1, n+1)])
    A = np.column_stack([x**5, x**4, x**3, x**2, x])

    # ② 普通最小二乘求解
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs           # shape (n,)

# 例子：七次多项式（n=7），常数项=0
x = np.linspace(-1, 1, 50)
y = 0.5*x #- 2*x**3 + 1.2*x**7 + 0.1#*np.random.randn(len(x))
coeffs = polyfit_no_const(x, y, n=4)
print(coeffs)   # a[0]→一次项, a[6]→七次项
coeffs = np.asarray(coeffs)
p = np.poly1d(np.append(coeffs, 0.0))

# 现在 p(x) = a4·x⁴ + a3·x³ + a2·x² + a1·x，常数项固定 0
print(p)            # ➜    a4 x^4 + a3 x^3 + a2 x^2 + a1 x