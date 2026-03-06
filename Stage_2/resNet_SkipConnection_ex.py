import torch
import torch.nn as nn

# ==============================================
# 📌 실습 목표
# 1. Skip Connection이 뭔지 직접 확인
# 2. 일반 블록 vs Residual 블록 비교
# 3. 그래디언트 소실 문제 직접 확인
# 4. ResNet으로 실제 학습해보기
# ==============================================


# ==============================================
# PART 1. Skip Connection 개념 직접 확인
# ==============================================
print("=" * 55)
print("PART 1. Skip Connection 개념")
print("=" * 55)

# 간단한 입력값
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
# shape: (1, 4) = 배치 1개, 특징 4개

# 일반 레이어 (Skip Connection 없음)
normal_layer = nn.Linear(4, 4)
# 입력 4개 → 출력 4개

# Skip Connection 적용
# F(x) = 레이어 통과한 결과
# y    = F(x) + x (원본 입력을 더함!)
with torch.no_grad():
    # torch.no_grad() = 그래디언트 계산 안 함
    # 개념 확인용이라 학습 필요 없음

    fx = normal_layer(x)
    # F(x): 레이어 통과한 결과

    y_normal = fx
    # 일반 방식: 레이어 결과만 사용

    y_skip = fx + x
    # Skip Connection: 레이어 결과 + 원본 입력!

print(f"원본 입력 x:          {x.numpy().round(3)}")
print(f"레이어 출력 F(x):     {fx.numpy().round(3)}")
print(f"일반 방식 y = F(x):   {y_normal.numpy().round(3)}")
print(f"Skip 방식 y = F(x)+x: {y_skip.numpy().round(3)}")
print()
print("핵심: Skip 방식은 원본 정보가 항상 살아있음!")
print()


# ==============================================
# PART 2. 일반 블록 vs Residual 블록
# ==============================================
print("=" * 55)
print("PART 2. 일반 블록 vs Residual 블록")
print("=" * 55)

# ----------------------------------------------
# 일반 블록: 레이어만 통과
# ----------------------------------------------
class NormalBlock(nn.Module):
    """
    일반적인 신경망 블록
    입력 → Conv → BN → ReLU → Conv → BN → 출력
    원본 입력을 전혀 사용하지 않음
    """

    def __init__(self, channels):
        super().__init__()

        # 첫 번째 Conv 레이어
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1
            # padding=1: 입출력 크기 동일하게 유지
        )
        self.bn1 = nn.BatchNorm2d(channels)
        # BatchNorm: 값의 범위 정규화

        # 두 번째 Conv 레이어
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 순전파: 그냥 레이어만 통과
        out = self.conv1(x)   # Conv
        out = self.bn1(out)   # BatchNorm
        out = self.relu(out)  # ReLU
        out = self.conv2(out) # Conv
        out = self.bn2(out)   # BatchNorm
        out = self.relu(out)  # ReLU
        return out
        # 원본 x는 완전히 버려짐!


# ----------------------------------------------
# Residual 블록: 레이어 + 원본 입력
# ----------------------------------------------
class ResidualBlock(nn.Module):
    """
    ResNet의 핵심 블록
    입력 → Conv → BN → ReLU → Conv → BN → (+원본) → ReLU → 출력
                                            ↑
                                    Skip Connection!
    """

    def __init__(self, channels):
        super().__init__()

        # 구조는 NormalBlock과 완전히 동일
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 원본 입력을 저장해둠 (Skip Connection용)
        identity = x
        # identity = "나중에 더할 원본 입력"

        # F(x) 계산: 레이어 통과
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 핵심! F(x) + x (원본 입력을 더함!)
        # 수식: y = F(x) + x
        out = out + identity

        # 더한 후 마지막 ReLU
        out = self.relu(out)

        return out


# 두 블록 비교
channels = 16
normal_block   = NormalBlock(channels)
residual_block = ResidualBlock(channels)

# 테스트 입력
test_input = torch.randn(1, channels, 8, 8)
# (배치 1, 채널 16, 8×8 Feature Map)

normal_output   = normal_block(test_input)
residual_output = residual_block(test_input)

print(f"입력 크기:          {list(test_input.shape)}")
print(f"일반 블록 출력:     {list(normal_output.shape)}")
print(f"Residual 블록 출력: {list(residual_output.shape)}")
print()
print("크기는 똑같지만 내부 계산이 다르다.")
print(f"일반 블록:    y = F(x)")
print(f"Residual 블록: y = F(x) + x  ← 원본 더함")
print()


# ==============================================
# PART 3. 그래디언트 소실 문제 직접 확인
# ==============================================
print("=" * 55)
print("PART 3. 그래디언트 소실 문제")
print("=" * 55)

# 레이어를 깊게 쌓을수록 그래디언트가 어떻게 변하나?

# 일반 깊은 모델 (Skip Connection 없음)
deep_normal = nn.Sequential(
    *[NormalBlock(16) for _ in range(5)]
    # NormalBlock을 5개 쌓기
    # * = 리스트를 풀어서 넣기
)

# ResNet 깊은 모델 (Skip Connection 있음)
deep_resnet = nn.Sequential(
    *[ResidualBlock(16) for _ in range(5)]
    # ResidualBlock을 5개 쌓기
)

# 같은 입력으로 테스트
x_test = torch.randn(
    1, 16, 8, 8,
    requires_grad=True
    # requires_grad=True: 그래디언트 추적
)

# 일반 모델 그래디언트 확인
out_normal = deep_normal(x_test)
loss_normal = out_normal.sum()
loss_normal.backward()
# backward(): 역전파로 그래디언트 계산

grad_normal = x_test.grad.abs().mean().item()
# 입력까지 전달된 그래디언트 크기

# 그래디언트 초기화 후 ResNet 확인
x_test.grad = None
# None으로 초기화 (이전 그래디언트 지우기)

out_resnet = deep_resnet(x_test)
loss_resnet = out_resnet.sum()
loss_resnet.backward()

grad_resnet = x_test.grad.abs().mean().item()

print(f"5개 레이어 통과 후 입력까지 전달된 그래디언트:")
print(f"  일반 모델:  {grad_normal:.6f}")
print(f"  ResNet:     {grad_resnet:.6f}")
print()
if grad_resnet > grad_normal:
    print("ResNet의 그래디언트가 더 크게 유지됨! ✅")
    print("→ 앞쪽 레이어까지 학습 신호가 잘 전달됨!")
else:
    print("이 실험에선 비슷하게 나왔지만")
    print("레이어가 100개+ 로 깊어지면 차이가 극명해짐!")
print()


# ==============================================
# PART 4. 실제 ResNet으로 학습
# ==============================================
print("=" * 55)
print("PART 4. ResNet으로 실제 학습")
print("=" * 55)

# ----------------------------------------------
# 간단한 ResNet 모델 정의
# ----------------------------------------------
class SimpleResNet(nn.Module):
    """
    간단한 ResNet 구조:
    Conv → ResBlock → ResBlock → ResBlock
    → GlobalAvgPool → FC → 분류
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # 첫 번째 Conv (이미지 → Feature Map)
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # 흑백(1채널) → 16채널 Feature Map
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # ResidualBlock 3개 쌓기
        self.res_blocks = nn.Sequential(
            ResidualBlock(16),  # 첫 번째 Residual 블록
            ResidualBlock(16),  # 두 번째 Residual 블록
            ResidualBlock(16),  # 세 번째 Residual 블록
        )

        # Global Average Pooling
        # Feature Map의 각 채널을 평균값 하나로 압축
        # (배치, 16, H, W) → (배치, 16, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 최종 분류 레이어
        self.classifier = nn.Linear(16, num_classes)
        # 16개 특징 → num_classes개 클래스 점수

    def forward(self, x):

        # 1. 첫 번째 Conv
        out = self.first_conv(x)
        # (1, 1, 28, 28) → (1, 16, 28, 28)

        # 2. Residual 블록들 통과
        out = self.res_blocks(out)
        # (1, 16, 28, 28) → (1, 16, 28, 28)
        # 크기는 유지, 특징만 변환

        # 3. Global Average Pooling
        out = self.global_avg_pool(out)
        # (1, 16, 28, 28) → (1, 16, 1, 1)

        # 4. Flatten (FC Layer 입력용)
        out = out.view(out.size(0), -1)
        # (1, 16, 1, 1) → (1, 16)
        # out.size(0) = 배치 크기
        # -1 = 나머지를 자동으로 계산

        # 5. 최종 분류
        out = self.classifier(out)
        # (1, 16) → (1, num_classes)

        return out


# ----------------------------------------------
# 모델 생성 및 학습
# ----------------------------------------------
model = SimpleResNet(num_classes=3)
# 3개 클래스 분류

criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss = Softmax + Cross-Entropy

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
    # Adam = SGD보다 더 똑똑한 Optimizer
    # 학습률을 자동으로 조절해줌!
)

# 학습 데이터 (28×28 흑백 이미지 30장)
x_train = torch.randn(30, 1, 28, 28)
# (30장, 흑백, 28픽셀, 28픽셀)

y_train = torch.randint(0, 3, (30,))
# 정답: 0, 1, 2 중 랜덤

print(f"{'Step':>4} | {'Loss':>8} | {'정확도':>8}")
print("-" * 30)

for step in range(50):

    model.train()
    # 학습 모드 (Dropout, BN 활성화)

    # ① 순전파
    pred = model(x_train)

    # ② Loss 계산
    loss = criterion(pred, y_train)

    # ③ 그래디언트 초기화
    optimizer.zero_grad()

    # ④ 역전파
    loss.backward()

    # ⑤ 가중치 업데이트
    optimizer.step()

    if (step + 1) % 10 == 0:
        correct = (pred.argmax(dim=1) == y_train).sum().item()
        accuracy = correct / len(y_train) * 100
        print(f"{step+1:>4} | "
              f"{loss.item():>8.4f} | "
              f"{accuracy:>7.1f}%")

print()


# ==============================================
# PART 5. Skip Connection 효과 시각화
# ==============================================
print("=" * 55)
print("PART 5. Skip Connection 핵심 정리")
print("=" * 55)

print("""
일반 신경망:
  입력 → [Conv1] → [Conv2] → 출력
          변환1      변환2
  문제: 레이어가 깊어질수록 그래디언트 소실!

ResNet (Skip Connection):
  입력 ──────────────────────→ (+) → 출력
    └→ [Conv1] → [Conv2] ──→
        변환1      변환2    원본 더하기!

  장점:
  ① 그래디언트가 원본 경로로 직접 전달됨
     역전파: ∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)
             항상 +1이 있어서 그래디언트 소실 없음!

  ② 레이어가 "변화량"만 학습하면 됨
     기존: 처음부터 완벽한 답을 학습
     ResNet: 원본에서 얼마나 수정할지만 학습
             (훨씬 쉬운 문제!)

  ③ 최악의 경우 F(x)=0 이어도
     y = 0 + x = x (원본 그대로 통과)
     → 레이어가 해를 끼치지 않음!
""")

print("수식 정리:")
print("  일반:  y = F(x)")
print("  ResNet: y = F(x) + x")
print()
print("그래디언트:")
print("  일반:  ∂L/∂x = ∂L/∂y × ∂F/∂x  (소실 가능)")
print("  ResNet: ∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)  (항상 1 이상!)")