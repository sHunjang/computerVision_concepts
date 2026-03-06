import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import copy

# ==============================================
# 📌 전이 학습 실습 목표
#
# 사전 학습된 ResNet18 모델을 가져와서
# MNIST 데이터에 맞게 재활용
#
# ResNet18:
# ImageNet (1,000,000장) 으로 미리 학습된 모델
# 1000개 클래스 분류 가능
# → 우리는 10개 클래스(0~9)로 교체해서 사용
# ==============================================


# ==============================================
# 📌 GPU 설정
# ==============================================
# torch.cuda.is_available()
# = 현재 컴퓨터에 CUDA GPU가 있는지 확인
# True  → GPU 사용 (cuda)
# False → CPU 사용 (cpu)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("=" * 55)
print("GPU 설정 확인")
print("=" * 55)
print(f"사용 장치: {device}")

if device.type == 'cuda':
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    # 0 = 첫 번째 GPU (여러 개일 경우 0, 1, 2...)
    print(f"GPU 메모리: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"GPU가 감지됨 빠르게 학습할 수 있음 🚀")
else:
    print(f"GPU 없음. CPU로 학습합니다.")
print()


# ==============================================
# STEP 1. 데이터 준비
# ==============================================
# ResNet18은 원래 224×224 컬러 이미지용으로 학습됨
# MNIST는 28×28 흑백 이미지
# → 크기와 채널을 맞춰줘야 함

transform = transforms.Compose([

    transforms.Resize((224, 224)),
    # 28×28 → 224×224 로 크기 키우기
    # ResNet18 입력 크기에 맞춤

    transforms.Grayscale(num_output_channels=3),
    # 흑백(1채널) → 컬러(3채널) 로 변환
    # ResNet18은 3채널 입력을 받음
    # 실제로는 3채널 전부 같은 값이지만
    # 모델 구조상 3채널이 필요

    transforms.ToTensor(),
    # 이미지 → Tensor 변환

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        # ImageNet 표준 정규화 값
        # ResNet18이 이 값으로 학습됐기 때문에
        # 똑같이 맞춰줘야 성능이 잘 나옴
    ),
])

train_dataset = datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False,
    download=True, transform=transform
)

# 빠른 실습을 위해 일부만 사용
# 전이 학습의 장점 = 적은 데이터로도 가능
train_dataset = Subset(train_dataset, range(2000))
# 훈련 데이터 2000장만 사용
test_dataset  = Subset(test_dataset,  range(500))
# 테스트 데이터 500장만 사용

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
test_loader  = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

print("=" * 55)
print("STEP 1. 데이터 준비")
print("=" * 55)
print(f"훈련 데이터: {len(train_dataset)}장 (전체 60,000장 중 일부)")
print(f"테스트 데이터: {len(test_dataset)}장")
print(f"이미지 크기: 28×28 → 224×224 (ResNet 입력 크기)")
print(f"채널: 흑백 1채널 → 컬러 3채널 (ResNet 입력 형식)")
print()


# ==============================================
# STEP 2. 사전 학습 모델 불러오기
# ==============================================
print("=" * 55)
print("STEP 2. 사전 학습 모델 (ResNet18)")
print("=" * 55)

resnet = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
    # IMAGENET1K_V1 = ImageNet으로 학습된 가중치
    # 이 가중치 덕분에 이미 특징 추출 능력이 있음
)

print(f"ResNet18 불러오기 완료")
print(f"마지막 FC 레이어: {resnet.fc}")
# Linear(512 → 1000): 1000개 클래스 분류용
print(f"원래 출력: 1000개 클래스 → 교체 예정: 10개 클래스")
print()

criterion = nn.CrossEntropyLoss()


# ==============================================
# 공통 학습/평가 함수 (GPU 적용)
# ==============================================
def train_and_evaluate(model, train_loader, test_loader,
                       optimizer, criterion, epochs=5,
                       method_name=""):
    """
    학습 + 평가 통합 함수

    GPU 핵심 3가지:
    ① model.to(device)     → 모델을 GPU로 이동
    ② images.to(device)    → 이미지를 GPU로 이동
    ③ labels.to(device)    → 정답을 GPU로 이동

    ⚠️ 모델과 데이터는 같은 장치(GPU or CPU)에 있어야 함
    모델은 GPU인데 데이터가 CPU면 오류 발생
    """

    # ⭐ GPU 핵심 1: 모델을 GPU로 이동
    # .to(device) = 모델의 모든 가중치를 GPU 메모리로 복사
    model = model.to(device)

    print(f"{method_name} 학습 중... (장치: {device})")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'정확도':>8}")
    print("-" * 30)

    accuracies = []

    for epoch in range(1, epochs + 1):

        # 학습
        model.train()
        for images, labels in train_loader:

            # ⭐ GPU 핵심 2, 3: 데이터를 GPU로 이동
            # DataLoader는 기본적으로 CPU에서 데이터를 불러옴
            # → 매 배치마다 GPU로 옮겨줘야 함
            images = images.to(device)
            # 이미지 텐서를 GPU 메모리로 이동
            labels = labels.to(device)
            # 정답 텐서를 GPU 메모리로 이동

            # 이제 모델(GPU)과 데이터(GPU)가 같은 곳에 있음
            pred = model(images)   # GPU에서 순전파 실행
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()        # GPU에서 역전파 실행
            optimizer.step()       # GPU에서 가중치 업데이트

        # 평가
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:

                # ⭐ 평가 데이터도 GPU로 이동
                images = images.to(device)
                labels = labels.to(device)

                pred = model(images)
                correct += (
                    pred.argmax(1) == labels
                ).sum().item()
                # .item() = GPU 텐서 → Python 숫자로 변환
                total += len(labels)

        accuracy = correct / total * 100
        accuracies.append(accuracy)
        print(f"{epoch:>5} | "
              f"{loss.item():>8.4f} | "
              f"{accuracy:>7.2f}%")

    print()
    return accuracies


# ==============================================
# STEP 3. Feature Extraction
# (Conv 전부 Freeze, FC만 학습)
# ==============================================
print("=" * 55)
print("STEP 3. 방법 1: Feature Extraction")
print("=" * 55)

model_fe = copy.deepcopy(resnet)
# deepcopy = 완전히 독립적인 복사본 생성
# (원본 resnet에 영향 없음)

# Conv 레이어 전부 Freeze
for param in model_fe.parameters():
    param.requires_grad = False
    # requires_grad=False = 역전파 시 이 가중치 건들이면 안됨.
    # = Freeze (얼리기)

# FC 레이어만 교체 - 이 레이어만 requires_grad=True 상태로 역전파 시 이 레이어의 가중치만 업데이트 되도록 설정
model_fe.fc = nn.Linear(512, 10)
# 새로 만든 레이어는 자동으로 학습 가능 상태
# 그래디언트는 FC 레이어에서만 계산되고 Conv 레이어까지는 전파되지 않음

# 학습 가능한 파라미터 수 확인
trainable_fe = sum(
    p.numel() for p in model_fe.parameters()
    if p.requires_grad
)
total_params = sum(
    p.numel() for p in model_fe.parameters()
)

print(f"전체 파라미터:    {total_params:,}개")
print(f"학습할 파라미터:  {trainable_fe:,}개")
print(f"Freeze 파라미터:  {total_params - trainable_fe:,}개")
print(f"학습 비율: {trainable_fe/total_params*100:.2f}%만 학습")
print()

optimizer_fe = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_fe.parameters()),
    # filter = requires_grad=True 인 파라미터만 선택
    # Freeze된 파라미터는 업데이트 안 함
    lr=0.001
)

fe_accuracies = train_and_evaluate(
    model_fe, train_loader, test_loader,
    optimizer_fe, criterion,
    method_name="Feature Extraction"
)


# ==============================================
# STEP 4. Fine-tuning
# (layer4 + FC 학습)
# ==============================================
print("=" * 55)
print("STEP 4. 방법 2: Fine-tuning")
print("=" * 55)

model_ft = copy.deepcopy(resnet)

# 일단 전부 Freeze
for param in model_ft.parameters():
    param.requires_grad = False

# layer4만 Freeze 해제
# ResNet18 구조:
# layer1(저수준) → layer2 → layer3 → layer4(고수준) → FC
# layer4 = 가장 복잡한 패턴 담당
# layer4만 Freeze해제해서 MNIST 데이터 특징에 맞게 고수준 패텀 감지 방식을 미세 조정
'''
왜 layer4 인가?
layer1: 선, 엣지 같은 단순한 패턴  ← 공통적이라 그대로 유지
layer2: 곡선, 모서리 패턴           ← 공통적이라 그대로 유지
layer3: 복잡한 조합 패턴            ← 공통적이라 그대로 유지
layer4: 최종 고수준 특징            ← 데이터마다 달라서 조정
'''
for param in model_ft.layer4.parameters():
    param.requires_grad = True
    # layer4만 학습 가능하게 Freeze 해제

# FC 교체
model_ft.fc = nn.Linear(512, 10)

trainable_ft = sum(
    p.numel() for p in model_ft.parameters()
    if p.requires_grad
)

print(f"전체 파라미터:    {total_params:,}개")
print(f"학습할 파라미터:  {trainable_ft:,}개")
print(f"학습 비율: {trainable_ft/total_params*100:.2f}%만 학습")
print(f"(layer4 + FC만 학습)")
print()

optimizer_ft = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_ft.parameters()),
    lr=0.0001
    # Fine-tuning은 학습률을 더 작게
    # 이미 좋은 가중치를 살살 조정하는 것이라서
    # 크게 바꾸면 오히려 망가질 수 있음
)

ft_accuracies = train_and_evaluate(
    model_ft, train_loader, test_loader,
    optimizer_ft, criterion,
    method_name="Fine-tuning"
)


# ==============================================
# STEP 5. 최종 비교
# ==============================================
print("=" * 55)
print("STEP 5. 최종 비교")
print("=" * 55)

print(f"{'방법':<20} | {'최종 정확도':>10} | "
      f"{'학습 파라미터':>15}")
print("-" * 55)
print(f"{'Feature Extraction':<20} | "
      f"{fe_accuracies[-1]:>9.2f}% | "
      f"{trainable_fe:>15,}")
print(f"{'Fine-tuning':<20} | "
      f"{ft_accuracies[-1]:>9.2f}% | "
      f"{trainable_ft:>15,}")
print()
print("전이 학습 장점 정리:")
print(f"  데이터: 60,000장 → 2,000장만 사용 (1/30로 줄임)")
print(f"  GPU 활용으로 학습 속도 대폭 향상")
print(f"  사전 학습 모델 덕분에 적은 데이터로도 높은 성능")
print()
print("GPU 핵심 3줄 요약:")
print(f"  ① device = torch.device('cuda' or 'cpu')")
print(f"  ② model.to(device)   → 모델을 GPU로")
print(f"  ③ data.to(device)    → 데이터를 GPU로 (매 배치마다)")