import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    """
    수식: softmax(z_i) = exp(z_i) / sum_j(exp(z_j))
    
    수치 안정성 트릭: exp(z_i - max(z)) / sum_j(exp(z_j - max(z)))
    → 이유: exp(1000)은 overflow 발생. max를 빼면 최댓값이 0이 되어 안전.
           수학적으로는 동일 (분자분모에 같은 상수 나눔)
    """
    # 수치 안정성을 위해 최댓값을 빼줌
    z_stable = z - np.max(z)           # overflow 방지
    exp_z = np.exp(z_stable)           # e^(z_i - max(z))
    return exp_z / np.sum(exp_z)       # 정규화

def cross_entropy_loss(y_pred_probs: np.ndarray, y_true_idx: int) -> float:
    """
    다중 클래스 Cross-Entropy Loss
    
    수식: L = -log(p_y)
    - p_y: 정답 클래스(y_true_idx)에 대한 예측 확률
    
    Args:
        y_pred_probs: softmax 출력 확률 벡터 (C,)
        y_true_idx:   정답 클래스 인덱스
    
    Returns:
        scalar loss 값
    """
    # 수치 안정성: log(0)  = -inf 방지를 위해 clipping
    epsilon = 1e-9
    prob_correct_class = y_pred_probs[y_true_idx]
    prob_correct_class = np.clip(prob_correct_class, epsilon, 1.0)
    
    # 핵심 수식: L = -log(p_y)
    loss = -np.log(prob_correct_class)
    return float(loss)

# 실습: 3개 클래스 분류 예제 (개 / 고양이 / 새)
logits = np.array([2.1, 0.5, -0.3])   # 모델의 raw 출력 (로짓)
probs = softmax(logits)

print("--- Softmax + Cross-Entropy 예제 ===")
print(f"로짓:     {logits}")
print(f"확률:     {probs.round(4)}")
print(f"확률 합:  {probs.sum():.6f}  <- 항상 1.0")
print()

true_class = 0   # 정답: 개 (index 0번째)
loss = cross_entropy_loss(probs, true_class)
print(f"정답 클래스 {true_class}의 예측 확률: {probs[true_class]:.4f}")
print(f"Cross-Entropy Loss: {loss:.4f}")
print()

# 완벽한 예측이면 Loss가 0에 가까워야 함
perfect_logits = np.array([100.0, 0.0, 0.0])   # 클래스 0을 매우 확신
perfect_probs = softmax(perfect_logits)
perfect_loss = cross_entropy_loss(perfect_probs, 0)

print(f"완벽한 예측의 Loss: {perfect_loss:.6f} <- 0에 가까움")