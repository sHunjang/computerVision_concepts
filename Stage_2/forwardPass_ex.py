import torch
import torch.nn as nn

# ==============================================
# 📌 Forward Pass란?
# 입력 이미지가 각 레이어를 통과해서
# 최종 예측값(확률)이 되는 과정
#
# 흐름:
# 이미지 → Conv → ReLU → Pooling → Flatten → FC → 예측값
# ==============================================


# ==============================================
# STEP 1. 입력 이미지 준비
# ==============================================
# torch.randn(N, C, H, W)
# N = 배치 크기 (한 번에 처리하는 이미지 수)
# C = 채널 수   (흑백=1, 컬러=3)
# H = 높이      (픽셀)
# W = 너비      (픽셀)
image = torch.randn(1, 1, 28, 28)
# → 28×28 흑백 이미지 1장

print("=" * 50)
print("STEP 1. 입력 이미지")
print("=" * 50)
print(f"이미지 크기: {image.shape}")
print()


# ==============================================
# STEP 2. Convolution Layer
# ==============================================
# 필터(패턴 감지기)가 이미지를 순환하면서
# 곱하고 더하기 → Feature Map 생성
#
# in_channels=1  : 입력 채널 수 (흑백이라 1)
# out_channels=8 : 필터 개수 → Feature Map 8장 생성
# kernel_size=3  : 필터 크기 (3×3)
# padding=1      : 테두리에 0 추가 → 출력 크기 유지
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1
)
feature_map = conv_layer(image)

print("=" * 50)
print("STEP 2. Convolution Layer 통과")
print("=" * 50)
print(f"입력 크기: {image.shape}")
print(f"출력 크기: {feature_map.shape}")
print(f"해석: 필터 8개 → Feature Map 8장 생성")
print()


# ==============================================
# STEP 3. ReLU
# ==============================================
# Feature Map의 음수값을 전부 0으로 만들기
# 음수 = "이 패턴이 없다" → 그냥 0으로 처리
relu = nn.ReLU()
feature_map_relu = relu(feature_map)

print("=" * 50)
print("STEP 3. ReLU 통과")
print("=" * 50)
print(f"입력 크기: {feature_map.shape}")
print(f"출력 크기: {feature_map_relu.shape}  (크기 그대로, 값만 바뀜)")
print(f"ReLU 전 최솟값: {feature_map.min().item():.4f}  ← 음수 있음")
print(f"ReLU 후 최솟값: {feature_map_relu.min().item():.4f}  ← 음수 전부 제거!")
print()


# ==============================================
# STEP 4. Max Pooling
# ==============================================
# 2×2 구역에서 가장 큰 값만 남기기
# → 크기를 절반으로 줄임
# → 계산량 감소 + 위치 불변성 확보
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool_layer(feature_map_relu)

print("=" * 50)
print("STEP 4. Max Pooling 통과")
print("=" * 50)
print(f"입력 크기: {feature_map_relu.shape}")
print(f"출력 크기: {pooled.shape}")
print(f"해석: 28×28 → 14×14 (크기 절반으로 줄어듦)")
print()


# ==============================================
# STEP 5. Flatten
# ==============================================
# 2D Feature Map → 1D 벡터로 펼치기
# FC Layer는 1D 벡터를 입력으로 받기 때문에 필요
# (8, 14, 14) → (1568,)
flatten = nn.Flatten()
flattened = flatten(pooled)

print("=" * 50)
print("STEP 5. Flatten 통과")
print("=" * 50)
print(f"입력 크기: {pooled.shape}")
print(f"출력 크기: {flattened.shape}")
print(f"해석: 2D 표 → 1D 벡터 (8 × 14 × 14 = {8*14*14})")
print()


# ==============================================
# STEP 6. Fully Connected Layer
# ==============================================
# 1D 벡터를 받아서 최종 분류 점수로 변환
# y = W·x + b (Stage 1에서 배운 행렬 곱!)
fc1 = nn.Linear(1568, 128)  # 1568 → 128 (특징 압축)
fc2 = nn.Linear(128, 10)    # 128 → 10  (클래스 점수)
# 사람이 직접 클래스 수를 정해서 코드에 반영해야함.
'''
예시. 숫자 분류 (MNIST)
사람이 정한 클래스:
0번 클래스 = 숫자 "0"
1번 클래스 = 숫자 "1"
2번 클래스 = 숫자 "2"
...
9번 클래스 = 숫자 "9"

→ 총 10개 클래스
→ 그래서 코드에서 출력이 10개였던 것

nn.Linear(128, 10)  ← 10이 클래스 수

*********************
클래스 수가 바뀌면 코드는 아래처럼 바꿔야함.

# 3종류 분류
nn.Linear(128, 3)   # 출력 3개

# 10종류 분류
nn.Linear(128, 10)  # 출력 10개

# 1000종류 분류
nn.Linear(128, 1000)  # 출력 1000개

모델이 하는 일:
정답과 예측이 다를 때 → Loss 계산
Loss를 줄이는 방향으로 → 가중치 업데이트
반복하다 보면 → 패턴을 스스로 발견
'''

relu2 = nn.ReLU()

fc1_out = relu2(fc1(flattened))  # 1568 → 128, ReLU 적용
logits  = fc2(fc1_out)           # 128 → 10 (최종 점수)

print("=" * 50)
print("STEP 6. Fully Connected Layer 통과")
print("=" * 50)
print(f"FC1: {flattened.shape} → {fc1_out.shape}")
print(f"FC2: {fc1_out.shape} → {logits.shape}")
print(f"최종 점수(로짓): {logits.detach().numpy().round(3)}")
print()


# ==============================================
# STEP 7. Softmax → 확률로 변환
# ==============================================
# 10개 점수(로짓) → 10개 확률
# 모든 값 0~1 사이, 합계 = 1.0
softmax = nn.Softmax(dim=1)
probabilities = softmax(logits)

print("=" * 50)
print("STEP 7. Softmax → 확률 변환")
print("=" * 50)
print(f"입력(로짓): {logits.detach().numpy().round(3)}")
print(f"출력(확률): {probabilities.detach().numpy().round(3)}")
print(f"확률 합계:  {probabilities.sum().item():.6f}  ← 항상 1.0!")
print(f"예측 클래스: {probabilities.argmax().item()}번")
print()


# ==============================================
# 📌 전체 흐름 한눈에 정리 (수정된 버전)
# ==============================================
print("=" * 50)
print("📌 Forward Pass 전체 흐름 요약")
print("=" * 50)
print(f"입력 이미지  {list(image.shape)}")
print(f"  ↓ Conv2d (필터 8개)")
print(f"Feature Map  {list(feature_map.shape)}")
print(f"  ↓ ReLU (음수 제거)")
print(f"Feature Map  {list(feature_map_relu.shape)}")
print(f"  ↓ MaxPool2d (크기 절반)")
print(f"Feature Map  {list(pooled.shape)}")
print(f"  ↓ Flatten (1D로 펼치기)")
print(f"벡터         {list(flattened.shape)}")
print(f"  ↓ Linear + ReLU")
print(f"벡터         {list(fc1_out.shape)}")
print(f"  ↓ Linear")
print(f"로짓         {list(logits.shape)}  (10개 클래스 점수)")
print(f"  ↓ Softmax")
print(f"확률         {list(probabilities.shape)}  (합계 = 1.0)")
print()
print(f"최종 예측: {probabilities.argmax().item()}번 클래스")
print(f"확신도:    {probabilities.max().item()*100:.1f}%")