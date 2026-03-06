import torch
import torch.nn as nn

# ==============================================
# 1. 모델 정의
# ==============================================
model = nn.Sequential(
    nn.Linear(4, 8),   # 입력 4개 → 출력 8개
    nn.ReLU(),         # 음수 제거
    nn.Linear(8, 3),   # 입력 8개 → 출력 3개 (3클래스)
)

# ==============================================
# 2. Loss 함수 정의
# CrossEntropyLoss = Softmax + Cross-Entropy
# 분류 문제의 표준 Loss
# ==============================================
criterion = nn.CrossEntropyLoss()

# ==============================================
# 3. Optimizer 정의
# SGD = Stochastic Gradient Descent
# lr = learning rate (학습률 η)
# ==============================================
optimizer = torch.optim.SGD(
    model.parameters(),  # 학습할 가중치들
    lr=0.01              # 한 걸음 크기
)

# ==============================================
# 4. 가짜 데이터 생성
# ==============================================
# 입력: 특징 4개짜리 데이터 5개
x = torch.randn(5, 4)
# (배치 5개, 특징 4개)

# 정답: 각 데이터의 클래스
y = torch.tensor([0, 1, 2, 0, 1])
# 클래스 0, 1, 2 중 하나

# ==============================================
# 5. 학습 루프
# ==============================================
for step in range(20):  # 20번 반복

    # ① 순전파: 모델에 데이터를 넣어 예측
    pred = model(x)
    # pred.shape = (5, 3): 5개 데이터, 각 3개 클래스 점수

    # ② Loss 계산: 예측 vs 정답 비교
    loss = criterion(pred, y)

    # ③ 그래디언트 초기화
    # ⚠️ 안 하면 이전 스텝의 그래디언트가 누적됨!
    optimizer.zero_grad()

    # ④ 역전파: 각 가중치의 그래디언트 자동 계산
    loss.backward()

    # ⑤ 가중치 업데이트
    # 수식: W = W - lr × gradient
    optimizer.step()

    # 5 스텝마다 출력
    if (step + 1) % 5 == 0:
        print(f"Step {step+1:2d} | Loss: {loss.item():.4f}")