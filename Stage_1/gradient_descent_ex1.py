import numpy as np

def gradient_descent_demo():
    """
    Gradient Descent 시각화 데모
    
    목표: f(x) = x^2 + 2x + 1 = (x+1)^2 의 최솟값 찾기
    최솟값: x = -1 (f(-1) = 0)
    
    수학:
    - f(x) = x^2 + 2x + 1
    - f'(x) = 2x + 2  ← 이것이 gradient
    - 업데이트: x_{t+1} = x_t - η * f'(x_t)
    """
    
    # 목적 함수 및 그래디언트 정의
    def f(x):
        return x**2 + 2*x + 1    # (x+1)^2
    
    def grad_f(x):
        return 2*x + 2  # f'(x) = 2x + 2
		    
    # 초기값 및 하이퍼파라미터
    x = 3.0      # 임의의 시작점
    eta = 0.1    # 학습률
    n_steps = 30 # 반복 횟수
    
    history = []
    
    for step in range(n_steps):
        fx = f(x)
        grad = grad_f(x)
        
        history.append({'step': step, 'x': x, 'f(x)': fx, 'grad': grad})
        
        # 핵심 업데이트 수식: θ_{t+1} = θ_t - η * ∇f(θ_t)
        x = x - eta * grad
    
    # 결과 출력
    print(f"{'Step':>4} | {'x':>8} | {'f(x)':>10} | {'gradient':>10}")
    print("-" * 45)
    for h in history[::5]:    # 5 스텝마다 출력
            print(f"{h['step']:>4} | {h['x']:>8.4f} | {h['f(x)']:>10.6f} | {h['grad']:>10.4f}")
    print(f"\n최종 x = {x:.6f} (정답: -1.0)")
    print(f"최종 f(x) = {f(x):.8f} (정답: 0.0)")

gradient_descent_demo()