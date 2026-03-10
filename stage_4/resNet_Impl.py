'''
ResNet 논문 구현 (Replication)
"Deep Residual Learning for Image Recognition"
He et al., 2015

논문 구조 그대로 구현:
① Residual Block
② ResNet-18 / ResNet-34
③ CIFAR-10으로 성능 확인
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ==============================================
# GPU 설정
# ==============================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"사용 장치: {device}\n")


# ==============================================
# PART 1. Residual Block 구현
# ==============================================
# 논문 핵심 수식:
# y = F(x, {Wi}) + x
# F(x) = Conv → BN → ReLU → Conv → BN
# ==============================================

class ResidualBlock(nn.Module):
    '''
    논문 Figure 2의 기본 Residual Block

    구조:
    x ──→ Conv → BN → ReLU → Conv → BN ──→ F(x)
    │                                         │
    └─────────── Skip Connection ─────────────┘
                                              ↓
                                         F(x) + x
                                              ↓
                                            ReLU

    논문 수식:
    y = F(x, {Wi}) + x
    '''

    expansion = 1
    # BasicBlock은 채널 수 변화 없음
    # (BottleneckBlock은 4배 확장, 나중에 설명)

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # F(x) 부분: Conv → BN → ReLU → Conv → BN
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,  # stride=2면 feature map 절반으로 줄임
            padding=1,
            bias=False       # BN을 쓸 때는 bias 불필요
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip Connection 처리
        # 논문 수식: y = F(x) + Ws*x
        # 채널 수나 크기가 달라질 때 Ws (1×1 Conv) 사용
        self.shortcut = nn.Sequential()
        # 기본: 그냥 x를 그대로 더함 (identity)

        if stride != 1 or in_channels != out_channels:
            # 입력과 출력 크기가 다를 때
            # 1×1 Conv로 차원 맞춰줌 (Projection Shortcut)
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,  # 1×1 Conv
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # F(x) 계산
        out = F.relu(self.bn1(self.conv1(x)))
        # Conv → BN → ReLU
        out = self.bn2(self.conv2(out))
        # Conv → BN (ReLU는 더하기 후에!)

        # 핵심: y = F(x) + x
        out = out + self.shortcut(x)
        # F(x) + Skip Connection

        out = F.relu(out)
        # 더한 후 ReLU

        return out


# ==============================================
# PART 2. ResNet 전체 구조 구현
# ==============================================
# 논문 Table 1의 구조 그대로 구현
# ==============================================

class ResNet(nn.Module):
    '''
    논문 Table 1 기준 ResNet 구현

    ResNet-18: [2, 2, 2, 2] blocks
    ResNet-34: [3, 4, 6, 3] blocks

    전체 구조:
    Conv1 (7×7, 64, stride=2)
    → MaxPool (3×3, stride=2)
    → Layer1 (64 channels)
    → Layer2 (128 channels, stride=2)
    → Layer3 (256 channels, stride=2)
    → Layer4 (512 channels, stride=2)
    → AvgPool
    → FC (num_classes)
    '''

    def __init__(self, block, num_blocks, num_classes=10):
        # num_classes=10: CIFAR-10 기준
        # ImageNet이면 1000으로 변경
        super().__init__()

        self.in_channels = 64
        # 현재 입력 채널 수 추적용

        # 첫 번째 Conv (논문: 7×7, 64, stride=2)
        # CIFAR-10은 이미지가 작아서 3×3, stride=1로 수정
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,   # 논문은 7×7, CIFAR용으로 3×3
            stride=1,        # 논문은 stride=2, CIFAR용으로 1
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # 논문: MaxPool (3×3, stride=2)
        # CIFAR-10은 이미지가 작아서 생략

        # 4개의 Layer 그룹
        # 각 그룹 = 여러 개의 Residual Block
        self.layer1 = self._make_layer(
            block, 64,  num_blocks[0], stride=1
        )
        # 채널: 64 → 64, 크기 유지

        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2
        )
        # 채널: 64 → 128, 크기 절반

        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2
        )
        # 채널: 128 → 256, 크기 절반

        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2
        )
        # 채널: 256 → 512, 크기 절반

        # Global Average Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 어떤 크기의 feature map이든 1×1로 만들어줌

        self.fc = nn.Linear(
            512 * block.expansion, num_classes
        )
        # 512 → 10 (CIFAR-10 클래스 수)


    def _make_layer(self, block, out_channels,
                    num_blocks, stride):
        '''
        여러 개의 Residual Block을 쌓는 함수

        첫 번째 블록: stride 적용 (크기 줄이기)
        나머지 블록: stride=1 (크기 유지)
        '''
        strides = [stride] + [1] * (num_blocks - 1)
        # [2, 1, 1, ...] 형태
        # 첫 번째만 stride 적용

        layers = []
        for s in strides:
            layers.append(
                block(self.in_channels, out_channels, s)
            )
            self.in_channels = out_channels * block.expansion
            # 다음 블록의 입력 채널 업데이트

        return nn.Sequential(*layers)


    def forward(self, x):
        # 입력: (N, 3, 32, 32) CIFAR-10 기준

        out = F.relu(self.bn1(self.conv1(x)))
        # (N, 64, 32, 32)

        out = self.layer1(out)
        # (N, 64, 32, 32)

        out = self.layer2(out)
        # (N, 128, 16, 16)

        out = self.layer3(out)
        # (N, 256, 8, 8)

        out = self.layer4(out)
        # (N, 512, 4, 4)

        out = self.avgpool(out)
        # (N, 512, 1, 1)

        out = out.view(out.size(0), -1)
        # (N, 512) Flatten

        out = self.fc(out)
        # (N, 10)

        return out


# ==============================================
# 논문 그대로 모델 생성 함수
# ==============================================

def ResNet18():
    '''
    논문 Table 1: ResNet-18
    [2, 2, 2, 2] BasicBlocks
    파라미터: 약 11M
    '''
    return ResNet(ResidualBlock, [2, 2, 2, 2])


def ResNet34():
    '''
    논문 Table 1: ResNet-34
    [3, 4, 6, 3] BasicBlocks
    파라미터: 약 21M
    '''
    return ResNet(ResidualBlock, [3, 4, 6, 3])


# ==============================================
# PART 3. 모델 구조 확인
# ==============================================

print("=" * 55)
print("PART 1. 모델 구조 확인")
print("=" * 55)

model_18 = ResNet18()
model_34 = ResNet34()

# 파라미터 수 계산
params_18 = sum(
    p.numel() for p in model_18.parameters()
)
params_34 = sum(
    p.numel() for p in model_34.parameters()
)

print(f"ResNet-18 파라미터: {params_18:,}개")
print(f"ResNet-34 파라미터: {params_34:,}개")

# 간단한 Forward Pass 테스트
test_input = torch.randn(4, 3, 32, 32)
# (배치 4장, RGB 3채널, 32×32)

with torch.no_grad():
    out_18 = model_18(test_input)
    out_34 = model_34(test_input)

print(f"\nForward Pass 테스트:")
print(f"입력 shape: {test_input.shape}")
print(f"ResNet-18 출력: {out_18.shape}")
# (4, 10): 4장 × 10개 클래스
print(f"ResNet-34 출력: {out_34.shape}")
print()


# ==============================================
# PART 4. Skip Connection 효과 확인
# ==============================================
# 논문 Figure 6 재현:
# 일반 CNN vs ResNet 학습 곡선 비교
# ==============================================

print("=" * 55)
print("PART 2. 일반 CNN vs ResNet 비교 준비")
print("=" * 55)

class PlainNet(nn.Module):
    '''
    Skip Connection 없는 일반 CNN
    논문에서 비교 대상으로 사용한 모델
    같은 구조지만 Skip Connection만 제거!
    '''
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(
                PlainBlock(self.in_channels, out_channels, s)
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class PlainBlock(nn.Module):
    '''
    Skip Connection 없는 일반 블록
    ResidualBlock과 동일하지만
    out = out + shortcut(x) 부분만 제거!
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        # Skip Connection 없음! F(x)만 사용
        return out


# ==============================================
# PART 5. CIFAR-10 데이터 준비
# ==============================================

print("=" * 55)
print("PART 3. CIFAR-10 데이터 준비")
print("=" * 55)

# 논문과 동일한 전처리
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # 논문: 데이터 증강 (랜덤 크롭)
    transforms.RandomHorizontalFlip(),
    # 논문: 좌우 반전 증강
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
        # CIFAR-10 표준 정규화 값
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test
)

train_loader = DataLoader(
    train_dataset, batch_size=128,
    # 논문: batch_size=128
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=128,
    shuffle=False,
    num_workers=0
)

print(f"훈련 데이터: {len(train_dataset):,}장")
print(f"테스트 데이터: {len(test_dataset):,}장")
print(f"클래스: {train_dataset.classes}")
print()


# ==============================================
# PART 6. 학습 함수
# ==============================================

def train_one_epoch(model, loader, optimizer,
                    criterion, device):
    '''한 epoch 학습'''
    model.train()
    total_loss = 0
    correct = total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total * 100


def evaluate(model, loader, criterion, device):
    '''테스트 정확도 평가'''
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            correct += (
                pred.argmax(1) == labels
            ).sum().item()
            total += len(labels)

    return correct / total * 100


# ==============================================
# PART 7. PlainNet vs ResNet 학습 비교
# ==============================================
# 논문 Figure 6 재현!
# ==============================================

print("=" * 55)
print("PART 4. PlainNet vs ResNet 학습 비교")
print("=" * 55)
print("논문 Figure 6 재현: Skip Connection 효과 확인!")
print()

EPOCHS = 10
# 논문은 160 epoch, 빠른 실습을 위해 10 epoch

# PlainNet (Skip Connection 없음)
plain_model = PlainNet([2, 2, 2, 2]).to(device)
plain_optimizer = torch.optim.SGD(
    plain_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
    # 논문과 동일한 하이퍼파라미터!
)

# ResNet-18 (Skip Connection 있음)
resnet_model = ResNet18().to(device)
resnet_optimizer = torch.optim.SGD(
    resnet_model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
    # 논문과 동일한 하이퍼파라미터!
)

# Learning Rate Scheduler
# 논문: epoch 80, 120에서 lr을 0.1배로 줄임
plain_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    plain_optimizer,
    milestones=[5, 8],
    # 간소화: epoch 5, 8에서 줄임
    gamma=0.1
)
resnet_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    resnet_optimizer,
    milestones=[5, 8],
    gamma=0.1
)

criterion = nn.CrossEntropyLoss()

print(f"{'Epoch':>5} | "
      f"{'Plain Loss':>10} | {'Plain Acc':>9} | "
      f"{'ResNet Loss':>11} | {'ResNet Acc':>10}")
print("-" * 60)

plain_accs  = []
resnet_accs = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # PlainNet 학습
    p_loss, _ = train_one_epoch(
        plain_model, train_loader,
        plain_optimizer, criterion, device
    )
    p_acc = evaluate(
        plain_model, test_loader, criterion, device
    )

    # ResNet 학습
    r_loss, _ = train_one_epoch(
        resnet_model, train_loader,
        resnet_optimizer, criterion, device
    )
    r_acc = evaluate(
        resnet_model, test_loader, criterion, device
    )

    plain_scheduler.step()
    resnet_scheduler.step()

    plain_accs.append(p_acc)
    resnet_accs.append(r_acc)

    elapsed = time.time() - start
    print(f"{epoch:>5} | "
          f"{p_loss:>10.4f} | {p_acc:>8.2f}% | "
          f"{r_loss:>11.4f} | {r_acc:>9.2f}%  "
          f"({elapsed:.0f}s)")

print()


# ==============================================
# PART 8. 최종 결과 비교
# ==============================================

print("=" * 55)
print("PART 5. 최종 결과 비교")
print("=" * 55)

print(f"PlainNet  최종 정확도: {plain_accs[-1]:.2f}%")
print(f"ResNet-18 최종 정확도: {resnet_accs[-1]:.2f}%")
print()
print(f"Skip Connection 효과: "
      f"+{resnet_accs[-1] - plain_accs[-1]:.2f}%")
print()

# 논문 결과와 비교
print("논문 결과 비교 (CIFAR-10, 110층 기준):")
print(f"  PlainNet:  6.61% 오류율")
print(f"  ResNet:    6.43% 오류율")
print(f"  → Skip Connection으로 성능 향상 확인!")
print()

# 파라미터 수 비교
plain_params = sum(
    p.numel() for p in plain_model.parameters()
)
resnet_params = sum(
    p.numel() for p in resnet_model.parameters()
)

print(f"파라미터 수 비교:")
print(f"  PlainNet:  {plain_params:,}개")
print(f"  ResNet-18: {resnet_params:,}개")
print(f"  → 거의 동일! (구조만 다름)")
print()
print("논문 구현(Replication) 완료! 🎉")