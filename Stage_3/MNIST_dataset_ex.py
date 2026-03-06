import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==============================================
# 📌 Stage 3 실습 목표
#
# Stage 2: 가짜 데이터로 개념 학습
# Stage 3: 진짜 데이터로 실전 학습!
#
# 전체 흐름:
# 데이터 준비 → 전처리 → 모델 정의
# → 학습 → 평가 → 저장/불러오기
# ==============================================


# ==============================================
# STEP 1. 데이터 전처리 정의 (transforms)
# ==============================================
# transforms.Compose = 여러 전처리를 순서대로 적용
# 마치 공장 컨베이어 벨트처럼!
#
# 원본 이미지 (PIL Image)
#     ↓ ToTensor
# (1, 28, 28) float32 [0, 1]
#     ↓ Normalize
# (1, 28, 28) float32 [-1, 1]  ← 모델 입력 준비!
transform = transforms.Compose([

    # ToTensor:
    # PIL 이미지 → PyTorch Tensor로 변환
    # 픽셀값 [0, 255] → [0, 1] 로 자동 변환
    # (H, W) → (C, H, W) 채널 추가
    transforms.ToTensor(),

    # Normalize:
    # 값의 범위를 통일 (과목별 점수 범위 통일!)
    # mean=평균, std=표준편차 (MNIST 전용 값)
    # 수식: x_norm = (x - 0.1307) / 0.3081
    # 결과: 값이 -1 ~ 1 사이로 맞춰짐
    transforms.Normalize(
        mean=(0.1307,),  # MNIST 전체 픽셀 평균
        std=(0.3081,)    # MNIST 전체 픽셀 표준편차
    ),
])

print("=" * 55)
print("STEP 1. 데이터 전처리 정의 완료")
print("=" * 55)
print("변환 순서:")
print("  원본 이미지 (PIL)")
print("  ↓ ToTensor  → Tensor [0, 1]")
print("  ↓ Normalize → Tensor [-1, 1]")
print()


# ==============================================
# STEP 2. 데이터셋 불러오기
# ==============================================
# datasets.MNIST = PyTorch가 제공하는 MNIST 데이터셋
# 처음 실행 시 인터넷에서 자동 다운로드!
# 이후엔 ./data 폴더에서 불러옴

# 훈련 데이터 (60,000장)
# = 교과서 + 기출문제집으로 공부하는 것
train_dataset = datasets.MNIST(
    root='./data',   # 데이터 저장 경로
    train=True,      # True = 훈련 데이터
    download=True,   # 없으면 자동 다운로드
    transform=transform  # 위에서 정의한 전처리 적용
)

# 테스트 데이터 (10,000장)
# = 수능처럼 한 번도 본 적 없는 문제로 실력 측정
test_dataset = datasets.MNIST(
    root='./data',
    train=False,     # False = 테스트 데이터
    download=True,
    transform=transform
)

print("=" * 55)
print("STEP 2. 데이터셋 불러오기")
print("=" * 55)
print(f"훈련 데이터: {len(train_dataset):,}장  ← 공부용")
print(f"테스트 데이터: {len(test_dataset):,}장 ← 시험용")

# 데이터 하나 확인
sample_image, sample_label = train_dataset[0]
print(f"이미지 크기: {sample_image.shape}")
# (1, 28, 28) = 흑백 1채널, 28×28 픽셀
print(f"정답 라벨 예시: {sample_label}번 (숫자 '{sample_label}')")
print()


# ==============================================
# STEP 3. DataLoader 생성
# ==============================================
# DataLoader = 데이터를 배치(묶음)로 나눠서 공급
# 6000명 손님을 64명씩 그룹으로 나눠서 서빙!

train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # 한 번에 64장씩 학습
                    # 너무 크면 메모리 부족
                    # 너무 작으면 학습 불안정
    shuffle=True,   # 매 epoch마다 순서 섞기
                    # 순서를 외워서 과적합되는 것 방지!
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,  # 테스트는 순서 고정
                    # 매번 같은 순서로 평가해야 공정함!
)

print("=" * 55)
print("STEP 3. DataLoader 생성")
print("=" * 55)
print(f"배치 크기: 64장")
print(f"훈련 배치 수: {len(train_loader)}개")
# 60,000 / 64 = 937개 배치
print(f"  (60,000 ÷ 64 = {len(train_loader)}번 학습 / epoch)")
print(f"테스트 배치 수: {len(test_loader)}개")
print()

# 배치 하나 확인
images, labels = next(iter(train_loader))
# iter() = DataLoader를 반복 가능하게 만들기
# next() = 다음 배치 하나 가져오기
print(f"배치 이미지 크기: {images.shape}")
# (64, 1, 28, 28) = 64장, 흑백, 28×28
print(f"배치 라벨 크기: {labels.shape}")
# (64,) = 64개 정답
print(f"배치 라벨 예시: {labels[:10].tolist()}")
# 처음 10개 정답
print()


# ==============================================
# STEP 4. 모델 정의
# ==============================================
class MNISTModel(nn.Module):
    """
    MNIST 분류 모델
    28×28 흑백 이미지 → 숫자 0~9 분류

    구조:
    Conv 블록1 → Conv 블록2 → Flatten → FC → 분류
    """

    def __init__(self):
        super().__init__()

        # ------------------------------------------
        # 특징 추출부: 이미지에서 패턴 찾기
        # Conv → BN → ReLU → Pooling 반복
        # ------------------------------------------
        self.features = nn.Sequential(

            # Conv 블록 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # 흑백(1채널) → 32채널 Feature Map
            # 28×28 크기 유지 (padding=1)
            nn.BatchNorm2d(32),
            # 32채널의 값 범위 정규화
            nn.ReLU(),
            # 음수 제거
            nn.MaxPool2d(2),
            # 28×28 → 14×14 (크기 절반!)

            # Conv 블록 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # 32채널 → 64채널 (더 많은 패턴 감지)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 14×14 → 7×7 (또 절반!)
        )

        # ------------------------------------------
        # 분류부: 찾은 패턴으로 숫자 판단
        # Flatten → FC → Dropout → FC
        # ------------------------------------------
        self.classifier = nn.Sequential(

            nn.Flatten(),
            # 2D Feature Map → 1D 벡터로 펼치기
            # (64, 7, 7) → (3136,)
            # 64 × 7 × 7 = 3136

            nn.Linear(64 * 7 * 7, 128),
            # 3136개 특징 → 128개로 압축
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 학습 중 50% 뉴런 랜덤하게 끔
            # 과적합 방지!

            nn.Linear(128, 10),
            # 128개 → 10개 (숫자 0~9)
        )

    def forward(self, x):
        # 순전파 정의
        x = self.features(x)
        # 이미지 → Feature Map 추출
        # (64, 1, 28, 28) → (64, 64, 7, 7)

        x = self.classifier(x)
        # Feature Map → 분류 점수
        # (64, 64, 7, 7) → (64, 10)

        return x


# ==============================================
# STEP 5. 학습 준비
# ==============================================
model     = MNISTModel()
criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss = Softmax + Cross-Entropy 한 번에!

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
    # Adam = SGD보다 똑똑한 Optimizer
    # 각 가중치마다 학습률을 자동으로 조절!
    # SGD보다 빠르게 수렴함
)

# Learning Rate Scheduler
# 산에서 내려올 때 가까워질수록 속도 줄이기!
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,  # 3 epoch마다 학습률 조정
    gamma=0.5     # 현재 학습률 × 0.5 (절반으로!)
    # epoch 1~3: lr = 0.001
    # epoch 4~6: lr = 0.0005
    # epoch 7~9: lr = 0.00025
)

print("=" * 55)
print("STEP 4-5. 모델 및 학습 준비")
print("=" * 55)

# 모델 파라미터 수 확인
# 파라미터 = 모델이 학습하는 가중치의 총 개수
total_params = sum(
    p.numel() for p in model.parameters()
)
# p.numel() = 해당 레이어의 파라미터 수
print(f"모델 파라미터 수: {total_params:,}개")
print(f"초기 학습률: {optimizer.param_groups[0]['lr']}")
print()


# ==============================================
# STEP 6. 학습 함수 정의
# ==============================================
def train_one_epoch(model, loader, criterion, optimizer):
    """
    1 epoch 학습 함수
    = 전체 훈련 데이터를 한 바퀴 다 학습
    """
    model.train()
    # 학습 모드 설정
    # Dropout 활성화, BN 배치 통계 사용

    total_loss = 0  # 전체 Loss 누적
    correct    = 0  # 맞춘 개수
    total      = 0  # 전체 개수

    for images, labels in loader:
        # 배치 하나씩 가져옴
        # images: (64, 1, 28, 28)
        # labels: (64,)

        # ① 순전파
        pred = model(images)
        # (64, 10): 64장 × 10개 클래스 점수

        # ② Loss 계산
        loss = criterion(pred, labels)

        # ③ 그래디언트 초기화
        optimizer.zero_grad()

        # ④ 역전파
        loss.backward()

        # ⑤ 가중치 업데이트
        optimizer.step()

        # 통계 누적
        total_loss += loss.item()
        correct += (
            pred.argmax(dim=1) == labels
        ).sum().item()
        # argmax(dim=1) = 가장 높은 점수의 클래스 번호
        total += len(labels)

    avg_loss = total_loss / len(loader)
    # 배치 수로 나눠서 평균 Loss
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ==============================================
# STEP 7. 평가 함수 정의
# ==============================================
def evaluate(model, loader, criterion):
    """
    모델 평가 함수
    = 테스트 데이터로 진짜 실력 측정 (수능!)
    """
    model.eval()
    # 평가 모드 설정
    # Dropout 비활성화, BN 고정 통계 사용

    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        # 그래디언트 계산 안 함
        # 테스트 중엔 가중치 업데이트 불필요
        # → 메모리 절약 + 속도 향상!

        for images, labels in loader:
            pred = model(images)
            loss = criterion(pred, labels)

            total_loss += loss.item()
            correct += (
                pred.argmax(dim=1) == labels
            ).sum().item()
            total += len(labels)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ==============================================
# STEP 8. 실제 학습 실행
# ==============================================
print("=" * 55)
print("STEP 6-8. 학습 시작!")
print("=" * 55)
print(f"{'Epoch':>5} | {'학습Loss':>8} | "
      f"{'학습정확도':>9} | {'테스트정확도':>11} | "
      f"{'학습률':>9}")
print("-" * 58)

best_accuracy = 0
# 지금까지 가장 좋은 테스트 정확도 기록

for epoch in range(1, 6):
    # 5 epoch 학습
    # = 전체 데이터를 5번 반복 학습

    # 1 epoch 학습
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer
    )

    # 테스트 데이터로 평가
    test_loss, test_acc = evaluate(
        model, test_loader, criterion
    )

    # 학습률 조정
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"{epoch:>5} | "
          f"{train_loss:>8.4f} | "
          f"{train_acc:>8.2f}% | "
          f"{test_acc:>10.2f}% | "
          f"{current_lr:>9.6f}")

    # 가장 좋은 성능일 때 모델 저장
    # = 게임 세이브 포인트!
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(
            model.state_dict(),
            'best_model.pth'
            # state_dict() = 모델의 모든 가중치
            # .pth = PyTorch 모델 저장 파일 형식
        )
        print(f"        → 최고 성능 갱신! 모델 저장 💾")

print()
print(f"최고 테스트 정확도: {best_accuracy:.2f}%")
print()


# ==============================================
# STEP 9. 저장된 모델 불러오기
# ==============================================
print("=" * 55)
print("STEP 9. 저장된 모델 불러오기")
print("=" * 55)

# 새로운 모델 객체 생성 (빈 모델)
loaded_model = MNISTModel()

# 저장된 가중치 불러오기
# = 게임 세이브 파일 불러오기!
loaded_model.load_state_dict(
    torch.load(
        'best_model.pth',
        weights_only=True
        # weights_only=True: 가중치만 불러옴
        # 보안상 권장되는 방법
    )
)
loaded_model.eval()

# 불러온 모델로 테스트
_, loaded_acc = evaluate(
    loaded_model, test_loader, criterion
)
print(f"원본 모델 정확도:     {best_accuracy:.2f}%")
print(f"불러온 모델 정확도:   {loaded_acc:.2f}%")
print(f"동일한가: "
      f"{'✅ 동일!' if abs(loaded_acc - best_accuracy) < 0.01 else '❌ 다름'}")
print()


# ==============================================
# STEP 10. 실제 예측 해보기
# ==============================================
print("=" * 55)
print("STEP 10. 실제 이미지 예측")
print("=" * 55)

# 테스트 이미지 5장 가져오기
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:5]
sample_labels = sample_labels[:5]

# 예측
loaded_model.eval()
with torch.no_grad():
    outputs = loaded_model(sample_images)
    # (5, 10): 5장 × 10개 클래스 점수

    probs = torch.softmax(outputs, dim=1)
    # 점수 → 확률로 변환 (합계 = 1.0)

    predicted = outputs.argmax(dim=1)
    # 가장 높은 확률의 클래스 번호

print(f"{'이미지':>5} | {'정답':>4} | "
      f"{'예측':>4} | {'확신도':>7} | {'결과':>4}")
print("-" * 35)

for i in range(5):
    correct_sign = '✅' if predicted[i] == sample_labels[i] else '❌'
    print(f"{i+1:>5} | "
          f"{sample_labels[i].item():>4} | "
          f"{predicted[i].item():>4} | "
          f"{probs[i].max().item()*100:>6.2f}% | "
          f"{correct_sign}")