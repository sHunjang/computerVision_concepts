import numpy as np

def activation_functions():
    """
    주요 활성화 함수 구현 및 비교
    
    각 함수의 수식:
    - ReLU:    max(0, x)
    - Sigmoid: 1 / (1 + exp(-x))
    - GELU:    x * Φ(x)  where Φ is standard normal CDF
    - Tanh:    (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    def relu(x: np.ndarray) -> np.ndarray:
        # max(0, x): 음수는 0으로, 양수는 그대로
        return np.maximum(0, x)
    
    def relu_grad(x: np.ndarray) -> np.ndarray:
        # 도함수: x>0이면 1, x<0이면 0
        return (x > 0).astype(np.float32)
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # 수치 안정성: 매우 음수 x에 대해 exp overflow 방지
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def sigmoid_grad(x: np.ndarray) -> np.ndarray:
        # 도함수: σ(x) * (1 - σ(x))
        s = sigmoid(x)
        return s * (1 - s)
    
    def gelu(x: np.ndarray) -> np.ndarray:
        # GELU(x) = x * Φ(x), Φ는 표준정규분포 CDF
        # 근사식 (실제 구현에서 자주 사용):
        # GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
        cdf = 0.5 * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        ))
        return x * cdf
    
    def tanh(x: np.ndarray) -> np.ndarray:
        # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return np.tanh(x)
    
    # 비교 테스트
    x = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    
    print(f"{'x':>6} | {'ReLU':>6} | {'Sigmoid':>8} | {'GELU':>8} | {'Tanh':>8}")
    print("-" * 50)
    for xi, r, s, g, t in zip(x, relu(x), sigmoid(x), gelu(x), tanh(x)):
        print(f"{xi:>6.1f} | {r:>6.3f} | {s:>8.4f} | {g:>8.4f} | {t:>8.4f}")
    
    print()
    print("=== 그래디언트 소실 문제 시연 ===")
    print("Sigmoid의 최대 그래디언트:", sigmoid_grad(np.array([0.0]))[0])
    print("→ 최대값이 0.25, 레이어 10개 쌓으면: 0.25^10 =", 0.25**10)
    print("→ 그래디언트가 너무 작아져 학습이 불가능해짐 (Gradient Vanishing)")
    print()
    print("ReLU의 양수 구간 그래디언트:", relu_grad(np.array([1.0]))[0])
    print("→ 1.0 그대로! 100개 레이어를 쌓아도 그래디언트가 1^100 = 1.0")

activation_functions()