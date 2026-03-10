'''
CLIP 논문 구현 (Replication)
"Learning Transferable Visual Models
 From Natural Language Supervision"
Radford et al., OpenAI 2021

구현 순서:
① Image Encoder (CNN 기반)
② Text Encoder (Transformer 기반)
③ CLIP 모델 (두 인코더 연결)
④ Contrastive Loss
⑤ 학습 파이프라인
⑥ Zero-shot 분류 테스트
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


# ==============================================
# GPU 설정
# ==============================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"사용 장치: {device}\n")


# ==============================================
# PART 1. Image Encoder
# ==============================================
# 논문: ResNet 또는 ViT 사용
# 실습: 간단한 CNN으로 구현
# ==============================================

class ImageEncoder(nn.Module):
    '''
    이미지를 임베딩 벡터로 변환
    논문: ResNet-50/101/ViT 사용
    실습: 간단한 CNN 사용

    입력: (batch, 3, 32, 32)
    출력: (batch, embed_dim)
    '''

    def __init__(self, embed_dim=128):
        super().__init__()

        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # (batch, 32, 16, 16)

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # (batch, 64, 8, 8)

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            # (batch, 128, 1, 1)
        )

        # 임베딩 차원으로 투영
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            # 논문: projection head로
            # 공통 임베딩 공간에 매핑!
        )

    def forward(self, x):
        '''
        Args:
            x: (batch, 3, 32, 32)
        Returns:
            (batch, embed_dim) L2 정규화된 벡터
        '''
        features = self.features(x)
        embedding = self.projection(features)

        # L2 정규화
        # 코사인 유사도 계산을 위해 필수!
        # 모든 벡터를 단위 구 위에 올려놓음
        return F.normalize(embedding, dim=-1)


# ==============================================
# PART 2. Text Encoder
# ==============================================
# 논문: Transformer 사용
# Transformer 구현에서 배운 것 그대로!
# ==============================================

class TextEncoder(nn.Module):
    '''
    텍스트를 임베딩 벡터로 변환
    논문: Transformer (GPT 스타일) 사용
    실습: 우리가 구현한 Transformer Encoder 활용

    입력: (batch, seq_len) 단어 인덱스
    출력: (batch, embed_dim)
    '''

    def __init__(self, vocab_size, embed_dim=128,
                 num_heads=4, num_layers=2,
                 max_len=32, dropout=0.1):
        super().__init__()

        d_model = embed_dim

        # 단어 임베딩
        self.embedding = nn.Embedding(
            vocab_size, d_model,
            padding_idx=0
        )

        # Positional Encoding
        self.pos_encoding = nn.Embedding(
            max_len, d_model
        )
        # 학습 가능한 위치 임베딩
        # (논문은 sin/cos 대신 학습 가능한 방식 사용)

        # Transformer Encoder 레이어들
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
            # (batch, seq, feature) 순서
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 임베딩 차원으로 투영
        self.projection = nn.Linear(d_model, embed_dim)

        self.max_len = max_len

    def forward(self, x):
        '''
        Args:
            x: (batch, seq_len) 단어 인덱스
        Returns:
            (batch, embed_dim) L2 정규화된 벡터
        '''
        batch_size, seq_len = x.shape

        # 패딩 마스크 생성
        # PAD(0)인 위치 무시
        padding_mask = (x == 0)
        # True인 위치를 무시

        # 단어 임베딩 + 위치 임베딩
        positions = torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, -1)

        token_emb = self.embedding(x)
        pos_emb   = self.pos_encoding(positions)
        emb = token_emb + pos_emb
        # (batch, seq_len, d_model)

        # Transformer 통과
        output = self.transformer(
            emb,
            src_key_padding_mask=padding_mask
        )
        # (batch, seq_len, d_model)

        # [CLS] 토큰 (첫 번째 토큰)으로 문장 표현
        cls_output = output[:, 0, :]
        # (batch, d_model)

        # 투영
        embedding = self.projection(cls_output)
        # (batch, embed_dim)

        # L2 정규화
        return F.normalize(embedding, dim=-1)


# ==============================================
# PART 3. CLIP 모델
# ==============================================
# 논문 핵심:
# Image Encoder + Text Encoder
# → 같은 임베딩 공간으로 매핑
# → Contrastive Loss로 학습
# ==============================================

class CLIP(nn.Module):
    '''
    논문 Figure 1 구조 그대로 구현

    학습:
    이미지-텍스트 짝을 같은 공간에 매핑
    → 짝인 것: 유사도 최대화
    → 짝 아닌 것: 유사도 최소화

    추론 (Zero-shot):
    이미지 임베딩과 텍스트 임베딩의
    코사인 유사도로 분류!
    '''

    def __init__(self, vocab_size, embed_dim=128,
                 num_heads=4, num_layers=2,
                 max_len=32):
        super().__init__()

        # 두 인코더
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder  = TextEncoder(
            vocab_size, embed_dim,
            num_heads, num_layers, max_len
        )

        # 논문: 학습 가능한 temperature 파라미터
        # logit_scale = log(1/τ)
        # 초기값: log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1 / 0.07)
        )
        # τ가 작을수록 유사도 분포가 날카로워짐
        # 학습하면서 최적값을 찾아감!

    def encode_image(self, images):
        '''이미지 → 임베딩'''
        return self.image_encoder(images)

    def encode_text(self, texts):
        '''텍스트 → 임베딩'''
        return self.text_encoder(texts)

    def forward(self, images, texts):
        '''
        Args:
            images: (batch, 3, H, W)
            texts:  (batch, seq_len)
        Returns:
            logits_per_image: (batch, batch)
            logits_per_text:  (batch, batch)
        '''

        # 각 인코더로 임베딩 추출
        image_features = self.encode_image(images)
        text_features  = self.encode_text(texts)
        # 둘 다 (batch, embed_dim), L2 정규화됨

        # Temperature 스케일링
        # 논문: logit_scale을 exp로 변환
        # clamp: 너무 커지지 않도록 제한
        logit_scale = self.logit_scale.exp().clamp(
            max=100
        )

        # 유사도 행렬 계산
        # (batch, embed_dim) × (embed_dim, batch)
        # = (batch, batch)
        logits_per_image = (
            logit_scale * image_features @ text_features.T
        )
        logits_per_text = logits_per_image.T
        # 전치 행렬 = 텍스트 → 이미지 방향

        return logits_per_image, logits_per_text


# ==============================================
# PART 4. Contrastive Loss
# ==============================================
# 논문 수식:
# L = (L_image + L_text) / 2
# 대각선(짝)의 확률을 최대화하는 Cross-Entropy
# ==============================================

class ContrastiveLoss(nn.Module):
    '''
    논문의 Symmetric Cross-Entropy Loss

    N×N 유사도 행렬에서
    대각선(i, i)이 가장 높아지도록!

    이미지 → 텍스트 방향: 각 행의 대각선
    텍스트 → 이미지 방향: 각 열의 대각선
    → 두 방향의 Loss 평균
    '''

    def forward(self, logits_per_image,
                logits_per_text):
        '''
        Args:
            logits_per_image: (batch, batch)
            logits_per_text:  (batch, batch)
        Returns:
            scalar loss
        '''
        batch_size = logits_per_image.shape[0]

        # 정답 레이블: 대각선 인덱스
        # [0, 1, 2, ..., N-1]
        labels = torch.arange(
            batch_size,
            device=logits_per_image.device
        )
        # i번째 이미지의 짝은 i번째 텍스트!

        # 이미지 → 텍스트 방향 Loss
        # 각 이미지가 올바른 텍스트를 선택
        loss_image = F.cross_entropy(
            logits_per_image, labels
        )

        # 텍스트 → 이미지 방향 Loss
        # 각 텍스트가 올바른 이미지를 선택
        loss_text = F.cross_entropy(
            logits_per_text, labels
        )

        # 두 방향의 평균
        # 논문: L = (L_I + L_T) / 2
        return (loss_image + loss_text) / 2


# ==============================================
# PART 5. 데이터셋 준비
# ==============================================
# CIFAR-10 이미지 + 텍스트 설명 쌍
# ==============================================

# CIFAR-10 클래스별 텍스트 템플릿
# 논문: "a photo of a {label}" 형식 사용!
CIFAR10_TEMPLATES = {
    0: [
        "a photo of an airplane",
        "an image of a flying airplane",
        "a picture of an aircraft in the sky",
    ],
    1: [
        "a photo of an automobile",
        "an image of a car on the road",
        "a picture of a vehicle",
    ],
    2: [
        "a photo of a bird",
        "an image of a bird flying",
        "a picture of a small bird",
    ],
    3: [
        "a photo of a cat",
        "an image of a cute cat",
        "a picture of a domestic cat",
    ],
    4: [
        "a photo of a deer",
        "an image of a deer in nature",
        "a picture of a wild deer",
    ],
    5: [
        "a photo of a dog",
        "an image of a cute dog",
        "a picture of a domestic dog",
    ],
    6: [
        "a photo of a frog",
        "an image of a green frog",
        "a picture of a frog on a leaf",
    ],
    7: [
        "a photo of a horse",
        "an image of a horse running",
        "a picture of a large horse",
    ],
    8: [
        "a photo of a ship",
        "an image of a large ship at sea",
        "a picture of a sailing vessel",
    ],
    9: [
        "a photo of a truck",
        "an image of a large truck",
        "a picture of a heavy vehicle",
    ],
}


class SimpleTokenizer:
    '''간단한 단어 토크나이저'''

    def __init__(self):
        self.word2idx = {
            '<PAD>': 0,
            '<CLS>': 1,
            '<UNK>': 2,
        }

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx

    def encode(self, sentence, max_len=32):
        tokens = ['<CLS>'] + sentence.lower().split()
        indices = [
            self.word2idx.get(t, 2)
            for t in tokens
        ]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

    @property
    def vocab_size(self):
        return len(self.word2idx)


class CIFAR10WithText(Dataset):
    '''
    CIFAR-10 이미지 + 텍스트 설명 쌍 데이터셋

    각 이미지마다 해당 클래스의
    텍스트 설명을 랜덤으로 하나 선택
    '''

    def __init__(self, cifar_dataset,
                 tokenizer, max_len=32):
        self.dataset   = cifar_dataset
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # 해당 클래스의 텍스트 중 랜덤 선택
        templates = CIFAR10_TEMPLATES[label]
        text = templates[
            np.random.randint(len(templates))
        ]

        # 텍스트 토크나이징
        text_tokens = torch.tensor(
            self.tokenizer.encode(text, self.max_len)
        )

        return image, text_tokens, label


# 토크나이저 구축
all_texts = [
    t
    for templates in CIFAR10_TEMPLATES.values()
    for t in templates
]

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(all_texts)

print("=" * 55)
print("PART 1. 데이터 준비")
print("=" * 55)

MAX_LEN    = 16
BATCH_SIZE = 256
EMBED_DIM  = 128

# CIFAR-10 데이터
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

cifar_train = datasets.CIFAR10(
    './data', train=True,
    download=True, transform=transform
)
cifar_test = datasets.CIFAR10(
    './data', train=False,
    download=True, transform=transform_test
)

train_dataset = CIFAR10WithText(
    cifar_train, tokenizer, MAX_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f"훈련 데이터:    {len(cifar_train):,}장")
print(f"테스트 데이터:  {len(cifar_test):,}장")
print(f"단어 사전 크기: {tokenizer.vocab_size}개")
print(f"텍스트 최대 길이: {MAX_LEN}")
print(f"임베딩 차원:    {EMBED_DIM}")
print()


# ==============================================
# PART 6. 모델 초기화
# ==============================================

print("=" * 55)
print("PART 2. 모델 구조 확인")
print("=" * 55)

model = CLIP(
    vocab_size  = tokenizer.vocab_size,
    embed_dim   = EMBED_DIM,
    num_heads   = 4,
    num_layers  = 2,
    max_len     = MAX_LEN,
).to(device)

criterion = ContrastiveLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
    # 논문: cosine annealing 사용
)

# 파라미터 수 확인
img_params  = sum(
    p.numel()
    for p in model.image_encoder.parameters()
)
text_params = sum(
    p.numel()
    for p in model.text_encoder.parameters()
)
total_params = sum(
    p.numel() for p in model.parameters()
)

print(f"Image Encoder 파라미터: {img_params:,}개")
print(f"Text  Encoder 파라미터: {text_params:,}개")
print(f"전체 파라미터:          {total_params:,}개")
print()

# Forward Pass 테스트
sample_images = torch.randn(4, 3, 32, 32).to(device)
sample_texts  = torch.randint(
    0, tokenizer.vocab_size, (4, MAX_LEN)
).to(device)

with torch.no_grad():
    logits_img, logits_txt = model(
        sample_images, sample_texts
    )

print(f"Forward Pass 테스트:")
print(f"이미지 입력: {sample_images.shape}")
print(f"텍스트 입력: {sample_texts.shape}")
print(f"유사도 행렬: {logits_img.shape}")
# (batch, batch) = N×N 행렬
print()


# ==============================================
# PART 7. Zero-shot 평가 함수
# ==============================================
# 논문의 핵심!
# 학습 안 한 방식으로 분류!
# ==============================================

def zero_shot_eval(model, test_dataset,
                   tokenizer, device,
                   num_samples=1000):
    '''
    논문 Figure 1 오른쪽 부분 구현

    Zero-shot 분류:
    ① 각 클래스의 텍스트 템플릿 임베딩
    ② 테스트 이미지 임베딩
    ③ 코사인 유사도로 가장 가까운 클래스 선택

    한 번도 이미지-레이블 쌍으로
    학습하지 않았지만 분류 가능!
    '''
    model.eval()

    # ① 각 클래스의 텍스트 임베딩 계산
    # 여러 템플릿의 평균을 사용
    # (논문: prompt ensemble)
    class_embeddings = []

    with torch.no_grad():
        for class_idx in range(10):
            templates = CIFAR10_TEMPLATES[class_idx]
            template_embeddings = []

            for template in templates:
                tokens = torch.tensor(
                    [tokenizer.encode(
                        template, MAX_LEN
                    )]
                ).to(device)

                emb = model.encode_text(tokens)
                template_embeddings.append(emb)

            # 여러 템플릿의 평균
            # 논문: prompt ensemble로 성능 향상!
            class_emb = torch.stack(
                template_embeddings
            ).mean(dim=0)
            class_emb = F.normalize(class_emb, dim=-1)
            class_embeddings.append(class_emb)

    # (10, embed_dim) = 10개 클래스 텍스트 임베딩
    class_embeddings = torch.cat(
        class_embeddings, dim=0
    )

    # ② 테스트 이미지 분류
    correct = 0
    total   = min(num_samples, len(test_dataset))

    indices = torch.randperm(
        len(test_dataset)
    )[:total]

    for idx in indices:
        image, true_label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            # 이미지 임베딩
            image_emb = model.encode_image(image)
            # (1, embed_dim)

            # ③ 코사인 유사도 계산
            # image_emb: (1, embed_dim)
            # class_embeddings: (10, embed_dim)
            similarities = (
                image_emb @ class_embeddings.T
            ).squeeze(0)
            # (10,) = 각 클래스와의 유사도

            # 가장 유사한 클래스 선택
            pred = similarities.argmax().item()

        if pred == true_label:
            correct += 1

    return correct / total * 100


# ==============================================
# PART 8. 학습 루프
# ==============================================

print("=" * 55)
print("PART 3. CLIP 학습")
print("=" * 55)
print("(매 Epoch 마다 Zero-shot 정확도 확인)")
print()

EPOCHS = 10

print(f"{'Epoch':>5} | {'Loss':>8} | "
      f"{'Zero-shot Acc':>13} | {'Temp(τ)':>8}")
print("-" * 45)

for epoch in range(1, EPOCHS + 1):

    # 학습
    model.train()
    total_loss   = 0
    num_batches  = 0

    for images, texts, labels in train_loader:
        images = images.to(device)
        texts  = texts.to(device)

        logits_img, logits_txt = model(images, texts)
        loss = criterion(logits_img, logits_txt)

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    scheduler.step()
    avg_loss = total_loss / num_batches

    # Zero-shot 평가
    zs_acc = zero_shot_eval(
        model, cifar_test, tokenizer,
        device, num_samples=500
    )

    # 현재 temperature 값
    tau = 1 / model.logit_scale.exp().item()

    print(f"{epoch:>5} | {avg_loss:>8.4f} | "
          f"{zs_acc:>12.2f}% | {tau:>8.4f}")

print()


# ==============================================
# PART 9. 최종 Zero-shot 분류 테스트
# ==============================================

print("=" * 55)
print("PART 4. 최종 Zero-shot 분류 테스트")
print("=" * 55)

# 클래스별 정확도 측정
model.eval()
class_names = [
    'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

# 클래스 임베딩 계산
with torch.no_grad():
    class_embeddings = []
    for class_idx in range(10):
        templates = CIFAR10_TEMPLATES[class_idx]
        embs = []
        for t in templates:
            tokens = torch.tensor(
                [tokenizer.encode(t, MAX_LEN)]
            ).to(device)
            emb = model.encode_text(tokens)
            embs.append(emb)
        class_emb = torch.stack(embs).mean(0)
        class_emb = F.normalize(class_emb, dim=-1)
        class_embeddings.append(class_emb)
    class_embeddings = torch.cat(
        class_embeddings, dim=0
    )

# 클래스별 정확도 측정
class_correct = [0] * 10
class_total   = [0] * 10

test_loader_eval = DataLoader(
    cifar_test,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

with torch.no_grad():
    for images, labels in test_loader_eval:
        images = images.to(device)

        image_embs = model.encode_image(images)
        # (batch, embed_dim)

        similarities = (
            image_embs @ class_embeddings.T
        )
        # (batch, 10)

        preds = similarities.argmax(dim=1)
        # (batch,)

        for pred, label in zip(
            preds.cpu(), labels
        ):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

print(f"{'클래스':<12} | {'정확도':>8} | 시각화")
print("-" * 45)

total_correct = sum(class_correct)
total_count   = sum(class_total)

for i, name in enumerate(class_names):
    acc = class_correct[i] / class_total[i] * 100
    bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
    print(f"{name:<12} | {acc:>7.1f}% | {bar}")

overall = total_correct / total_count * 100
print("-" * 45)
print(f"{'전체 정확도':<12} | {overall:>7.1f}%")
print()


# ==============================================
# PART 10. 임베딩 공간 시각화
# ==============================================
# 이미지와 텍스트가 같은 공간에 매핑됐는지 확인!

print("=" * 55)
print("PART 5. 임베딩 유사도 행렬")
print("=" * 55)
print("텍스트 임베딩 간 코사인 유사도")
print("(값이 클수록 비슷한 의미)")
print()

with torch.no_grad():
    # 클래스 간 유사도 행렬
    sim_matrix = (
        class_embeddings @ class_embeddings.T
    ).cpu().numpy()

# 상위 5개 클래스만 출력
top5 = ['airplane', 'automobile',
        'bird', 'cat', 'dog']
top5_idx = [0, 1, 2, 3, 5]

print(f"{'':>12}", end="")
for name in top5:
    print(f"{name:>12}", end="")
print()
print("-" * (12 + 12 * len(top5)))

for i, name in zip(top5_idx, top5):
    print(f"{name:>12}", end="")
    for j in top5_idx:
        val = sim_matrix[i][j]
        print(f"{val:>12.3f}", end="")
    print()

print()
print("CLIP 구현(Replication) 완료! 🎉")
print()
print("논문 핵심 구조 구현 완료:")
print("  ✅ Image Encoder (CNN)")
print("  ✅ Text Encoder (Transformer)")
print("  ✅ Contrastive Loss")
print("  ✅ Temperature Parameter (τ)")
print("  ✅ Zero-shot 분류")
print("  ✅ Prompt Ensemble")
print("  ✅ 임베딩 공간 시각화")