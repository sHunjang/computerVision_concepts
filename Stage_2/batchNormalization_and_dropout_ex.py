import torch
import torch.nn as nn

# ==============================================
# 📌 실습 목표
# 1. Batch Normalization이 값을 어떻게 바꾸는지 확인
# 2. Dropout이 뉴런을 어떻게 끄는지 확인
# 3. 둘 다 적용한 모델로 학습해보기
# ==============================================


# ==============================================
# PART 1. Batch Normalization 직접 확인
# ==============================================
print("=" * 55)
print("PART 1. Batch Normalization")
print("=" * 55)

# 📌 문제 상황: 값의 범위가 제각각인 데이터
# 수학 점수: 10~50점 (범위 작음)
# 영어 점수: 60~100점 (범위 큼)
# → 그냥 학습하면 영어 점수만 중요해짐!
before_bn = torch.tensor([
    [10.0, 80.0],   # 학생 1: 수학 10점, 영어 80점
    [20.0, 90.0],   # 학생 2: 수학 20점, 영어 90점
    [30.0, 70.0],   # 학생 3: 수학 30점, 영어 70점
    [40.0, 100.0],  # 학생 4: 수학 40점, 영어 100점
])
# shape: (4명, 2과목)

print("Batch Norm 적용 전:")
print(f"  수학 점수 범위: {before_bn[:, 0].min():.0f} ~ "
      f"{before_bn[:, 0].max():.0f}")
print(f"  영어 점수 범위: {before_bn[:, 1].min():.0f} ~ "
      f"{before_bn[:, 1].max():.0f}")
print(f"  → 범위가 달라서 영어만 중요해질 수 있음!\n")

# Batch Normalization 적용
# num_features=2 : 입력 특징 수 (2과목)
bn = nn.BatchNorm1d(num_features=2)

# 학습 모드에서 적용
bn.train()
after_bn = bn(before_bn)

print("Batch Norm 적용 후:")
print(f"  수학 점수 범위: {after_bn[:, 0].min().item():.3f} ~ "
      f"{after_bn[:, 0].max().item():.3f}")
print(f"  영어 점수 범위: {after_bn[:, 1].min().item():.3f} ~ "
      f"{after_bn[:, 1].max().item():.3f}")
print(f"  → 두 과목 모두 비슷한 범위로 맞춰짐! ✅")

print()
print("수식 확인: x̂ = (x - 평균) / 표준편차")
print(f"  수학 평균: {before_bn[:, 0].mean():.1f}")
print(f"  수학 표준편차: {before_bn[:, 0].std():.1f}")
print(f"  수학 첫 번째 값 정규화: "
      f"({before_bn[0,0].item():.0f} - "
      f"{before_bn[:,0].mean().item():.1f}) / "
      f"{before_bn[:,0].std().item():.1f} = "
      f"{after_bn[0,0].item():.3f}")
print()


# ==============================================
# PART 2. Dropout 직접 확인
# ==============================================
print("=" * 55)
print("PART 2. Dropout")
print("=" * 55)

# 뉴런 10개짜리 레이어
neurons = torch.ones(1, 10)
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 전부 1로 채워서 어떤 뉴런이 꺼지는지 확인

# Dropout 설정: 50% 확률로 뉴런을 끔
dropout = nn.Dropout(p=0.5)

print(f"원본 뉴런:   {neurons.tolist()}")
print()

# 학습 모드: Dropout 활성화
dropout.train()
print("[ 학습 모드 - Dropout 활성화 ]")
for i in range(3):
    # 매번 랜덤하게 다른 뉴런이 꺼짐!
    result = dropout(neurons)
    print(f"  시도 {i+1}: {result.tolist()}")
    # 꺼진 뉴런 = 0
    # 켜진 뉴런 = 2.0 (1/(1-0.5)=2 로 스케일 보정)

print()

# 평가 모드: Dropout 비활성화
dropout.eval()
print("[ 평가 모드 - Dropout 비활성화 ]")
for i in range(3):
    result = dropout(neurons)
    print(f"  시도 {i+1}: {result.tolist()}")
    # 전부 1.0 그대로! (Dropout 꺼짐)

print()
print("핵심: 학습 중엔 랜덤하게 끄고, 테스트 중엔 전부 켜기!")
print()


# ==============================================
# PART 3. Batch Norm + Dropout 적용한 모델
# ==============================================
print("=" * 55)
print("PART 3. 실제 모델에 적용")
print("=" * 55)

# ----------------------------------------------
# 모델 A: 기본 모델 (BN, Dropout 없음)
# ----------------------------------------------
model_basic = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

# ----------------------------------------------
# 모델 B: BN + Dropout 적용 모델
# ----------------------------------------------
model_improved = nn.Sequential(

    nn.Linear(4, 16),
    # Batch Norm: 16개 뉴런의 값을 정규화
    # Conv 다음엔 BatchNorm2d
    # Linear 다음엔 BatchNorm1d
    nn.BatchNorm1d(16),
    nn.ReLU(),
    # Dropout: 학습 중 30% 뉴런을 랜덤하게 끔
    nn.Dropout(p=0.3),

    nn.Linear(16, 8),
    nn.BatchNorm1d(8),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(8, 3),
)

# ----------------------------------------------
# 두 모델 비교 학습
# ----------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer_basic = torch.optim.SGD(
    model_basic.parameters(), lr=0.1
)
optimizer_improved = torch.optim.SGD(
    model_improved.parameters(), lr=0.1
)

# 학습 데이터
x_train = torch.randn(20, 4)
# 20개 데이터, 특징 4개
y_train = torch.randint(0, 3, (20,))
# 정답: 0, 1, 2 중 랜덤

print(f"{'Step':>4} | {'기본 Loss':>10} | {'개선 Loss':>10}")
print("-" * 35)

for step in range(50):

    # 기본 모델 학습
    model_basic.train()
    pred_basic = model_basic(x_train)
    loss_basic = criterion(pred_basic, y_train)
    optimizer_basic.zero_grad()
    loss_basic.backward()
    optimizer_basic.step()

    # 개선 모델 학습
    model_improved.train()
    pred_improved = model_improved(x_train)
    loss_improved = criterion(pred_improved, y_train)
    optimizer_improved.zero_grad()
    loss_improved.backward()
    optimizer_improved.step()

    # 10 스텝마다 출력
    if (step + 1) % 10 == 0:
        print(f"{step+1:>4} | "
              f"{loss_basic.item():>10.4f} | "
              f"{loss_improved.item():>10.4f}")

print()


# ==============================================
# PART 4. train() vs eval() 모드 차이
# ==============================================
print("=" * 55)
print("PART 4. train() vs eval() 모드")
print("=" * 55)

x_test = torch.randn(5, 4)

# 학습 모드: Dropout 활성화, BN 배치 통계 사용
model_improved.train()
out_train1 = model_improved(x_test)
out_train2 = model_improved(x_test)

# 평가 모드: Dropout 비활성화, BN 고정 통계 사용
model_improved.eval()
with torch.no_grad():
    # torch.no_grad(): 그래디언트 계산 안 함
    # 테스트 중엔 그래디언트 필요 없으니까
    # 메모리 절약 + 속도 향상!
    out_eval1 = model_improved(x_test)
    out_eval2 = model_improved(x_test)

print("[ 학습 모드 - 매번 다른 결과 (Dropout 때문에) ]")
print(f"  1번째 실행: {out_train1[0].detach().numpy().round(3)}")
print(f"  2번째 실행: {out_train2[0].detach().numpy().round(3)}")
print(f"  → 값이 다름! (랜덤하게 뉴런이 꺼지기 때문)")
print()
print("[ 평가 모드 - 매번 같은 결과 (Dropout 꺼짐) ]")
print(f"  1번째 실행: {out_eval1[0].detach().numpy().round(3)}")
print(f"  2번째 실행: {out_eval2[0].detach().numpy().round(3)}")
print(f"  → 값이 같음! (Dropout 비활성화) ✅")