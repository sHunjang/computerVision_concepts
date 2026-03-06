import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ==============================================
# 📌 성능 평가 실습 목표
#
# 단순 정확도(%)만 보는 것에서 벗어나
# 혼동 행렬로 모델이 어디서 틀리는지 분석!
#
# 평가 지표:
# ① 혼동 행렬  → 클래스별 오류 패턴
# ② Precision  → 맞다고 한 것 중 실제로 맞은 비율
# ③ Recall     → 실제 맞는 것 중 맞다고 한 비율
# ④ F1 Score   → Precision + Recall 균형 지표
# ==============================================


# ==============================================
# 한글 폰트 설정 (그래프에 한글 표시용)
# ==============================================
matplotlib.rcParams['axes.unicode_minus'] = False
# 마이너스 기호 깨짐 방지

plt.rcParams['font.family'] = 'DejaVu Sans'
# 기본 폰트 사용 (한글 대신 영어로 표시)


# ==============================================
# GPU 설정
# ==============================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"사용 장치: {device}\n")


# ==============================================
# STEP 1. 데이터 준비
# ==============================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    ),
])

# 테스트 데이터 전체 사용 (10,000장)
test_dataset = datasets.MNIST(
    root='./data', train=False,
    download=True, transform=transform
)

# 훈련 데이터도 준비 (모델 학습용)
train_dataset = datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False
    # shuffle=False: 순서 고정
    # 혼동 행렬 분석 시 순서가 중요하지 않지만
    # 재현성을 위해 고정!
)

print("=" * 55)
print("STEP 1. 데이터 준비")
print("=" * 55)
print(f"훈련 데이터: {len(train_dataset):,}장")
print(f"테스트 데이터: {len(test_dataset):,}장")
print()


# ==============================================
# STEP 2. 모델 정의 및 학습
# ==============================================
class MNISTModel(nn.Module):
    """
    Stage 3에서 만든 MNIST 분류 모델
    그대로 재사용!
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 28×28 → 14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 14×14 → 7×7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 모델 학습
print("=" * 55)
print("STEP 2. 모델 학습")
print("=" * 55)

model     = MNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"{'Epoch':>5} | {'Loss':>8} | {'정확도':>8}")
print("-" * 30)

for epoch in range(1, 6):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 간단한 정확도 확인
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred   = model(images)
            correct += (pred.argmax(1) == labels).sum().item()
            total   += len(labels)

    print(f"{epoch:>5} | "
          f"{loss.item():>8.4f} | "
          f"{correct/total*100:>7.2f}%")

print()


# ==============================================
# STEP 3. 예측값 수집
# ==============================================
# 혼동 행렬을 만들려면
# 전체 테스트 데이터의 정답과 예측을 모아야 함!

print("=" * 55)
print("STEP 3. 전체 예측값 수집")
print("=" * 55)

model.eval()

all_preds  = []  # 모델 예측값 저장
all_labels = []  # 실제 정답 저장
all_probs  = []  # 각 클래스 확률 저장

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # (64, 10): 64장 × 10개 클래스 점수

        probs = torch.softmax(outputs, dim=1)
        # 점수 → 확률로 변환

        preds = outputs.argmax(dim=1)
        # 가장 높은 확률의 클래스 번호

        # CPU로 옮겨서 numpy로 변환
        # GPU 텐서는 numpy로 바로 변환 불가!
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 리스트 → numpy 배열로 변환
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

print(f"수집된 예측값: {len(all_preds):,}개")
print(f"수집된 정답:   {len(all_labels):,}개")
print(f"전체 정확도:   {accuracy_score(all_labels, all_preds)*100:.2f}%")
print()


# ==============================================
# STEP 4. 혼동 행렬 계산 및 시각화
# ==============================================
print("=" * 55)
print("STEP 4. 혼동 행렬")
print("=" * 55)

# 혼동 행렬 계산
cm = confusion_matrix(all_labels, all_preds)
# 행 = 실제 정답, 열 = 모델 예측
# cm[i][j] = 정답이 i인데 j로 예측한 횟수

print("혼동 행렬 (숫자로 보기):")
print(f"{'':>4}", end="")
for i in range(10):
    print(f"{'pred_'+str(i):>8}", end="")
print()

for i in range(10):
    print(f"true_{i}", end="")
    for j in range(10):
        if i == j:
            print(f"{cm[i][j]:>8}", end="")
            # 대각선 (맞춘 것)
        else:
            val = cm[i][j]
            if val > 0:
                print(f"{val:>8}", end="")
                # 틀린 것 (0보다 크면 표시)
            else:
                print(f"{'·':>8}", end="")
                # 0이면 점으로 표시
    print()

print()

# 혼동 행렬 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: 실제 개수
sns.heatmap(
    cm,
    annot=True,       # 각 칸에 숫자 표시
    fmt='d',          # 정수로 표시
    cmap='Blues',     # 파란색 계열 색상
    xticklabels=range(10),  # x축: 예측 클래스
    yticklabels=range(10),  # y축: 실제 클래스
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix (Count)', fontsize=14)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# 오른쪽: 퍼센트로 표시
cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
# axis=1 = 행 방향으로 합계 (각 클래스별 정규화)
# keepdims=True = 행렬 형태 유지

sns.heatmap(
    cm_percent,
    annot=True,
    fmt='.1f',        # 소수점 1자리
    cmap='Blues',
    xticklabels=range(10),
    yticklabels=range(10),
    ax=axes[1]
)
axes[1].set_title('Confusion Matrix (%)', fontsize=14)
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
# 이미지로 저장
plt.show()
print("혼동 행렬 이미지 저장: confusion_matrix.png")
print()


# ==============================================
# STEP 5. 클래스별 상세 분석
# ==============================================
print("=" * 55)
print("STEP 5. 클래스별 상세 분석")
print("=" * 55)

# classification_report:
# 클래스별 Precision, Recall, F1 한 번에!
report = classification_report(
    all_labels, all_preds,
    target_names=[f"숫자 {i}" for i in range(10)]
)
print(report)


# ==============================================
# STEP 6. 가장 많이 틀린 케이스 분석
# ==============================================
print("=" * 55)
print("STEP 6. 자주 헷갈리는 클래스 쌍")
print("=" * 55)

# 대각선 제외하고 가장 큰 값 찾기
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
# 대각선을 0으로 만들어서 틀린 것만 분석

# 상위 5개 오류 찾기
errors = []
for i in range(10):
    for j in range(10):
        if i != j and cm_no_diag[i][j] > 0:
            errors.append((cm_no_diag[i][j], i, j))

errors.sort(reverse=True)
# 오류 횟수 내림차순 정렬

print(f"{'순위':>4} | {'실제':>6} | {'예측':>6} | {'횟수':>6} | 분석")
print("-" * 50)

for rank, (count, true, pred) in enumerate(errors[:5], 1):
    print(f"{rank:>4} | "
          f"숫자 {true:>2} | "
          f"숫자 {pred:>2} | "
          f"{count:>4}번 | "
          f"{true}을 {pred}로 헷갈림")

print()


# ==============================================
# STEP 7. 클래스별 정확도 시각화
# ==============================================
# 각 숫자별로 얼마나 잘 맞추는지 막대 그래프

class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
# diagonal() = 대각선 값 (맞춘 것)
# sum(axis=1) = 각 행의 합계 (전체 개수)

plt.figure(figsize=(10, 5))
bars = plt.bar(
    range(10),
    class_accuracy,
    color=['#2196F3' if acc >= 99 else
           '#FF9800' if acc >= 97 else
           '#F44336'
           for acc in class_accuracy]
    # 99% 이상: 파란색 (우수)
    # 97% 이상: 주황색 (보통)
    # 97% 미만: 빨간색 (부족)
)

# 막대 위에 정확도 숫자 표시
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f'{acc:.1f}%',
        ha='center', va='bottom', fontsize=10
    )

plt.title('Accuracy per Class', fontsize=14)
plt.xlabel('Digit Class')
plt.ylabel('Accuracy (%)')
plt.xticks(range(10), [f'Digit {i}' for i in range(10)])
plt.ylim(95, 101)
# y축 범위 설정 (차이가 잘 보이게)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("클래스별 정확도 이미지 저장: class_accuracy.png")
print()


# ==============================================
# STEP 8. 최종 성능 요약
# ==============================================
print("=" * 55)
print("STEP 8. 최종 성능 요약")
print("=" * 55)

print(f"전체 정확도:  {accuracy_score(all_labels, all_preds)*100:.2f}%")
print()
print(f"{'클래스':>6} | {'정확도':>8} | {'맞춘수':>8} | {'전체수':>8}")
print("-" * 40)

for i in range(10):
    correct_i = cm[i][i]
    # 대각선 = 맞춘 수
    total_i   = cm[i].sum()
    # 해당 행의 합계 = 전체 수
    acc_i     = correct_i / total_i * 100

    status = "✅" if acc_i >= 99 else "🔺" if acc_i >= 97 else "❌"
    print(f"숫자 {i:>2}  | "
          f"{acc_i:>7.2f}% | "
          f"{correct_i:>8} | "
          f"{total_i:>8}  {status}")