'''
Transformer 논문 구현 (Replication)
"Attention Is All You Need"
Vaswani et al., 2017

구현 순서:
① Scaled Dot-Product Attention
② Multi-Head Attention
③ Positional Encoding
④ Encoder Block (Attention + FFN + Add&Norm)
⑤ Transformer Encoder 전체
⑥ 감정 분류 태스크로 검증
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ==============================================
# GPU 설정
# ==============================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"사용 장치: {device}\n")


# ==============================================
# PART 1. Scaled Dot-Product Attention
# ==============================================
# 논문 핵심 수식:
# Attention(Q, K, V) = softmax(QK^T / √d_k) V
# ==============================================

class ScaledDotProductAttention(nn.Module):
    '''
    논문 Section 3.2.1

    수식:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Q: Query  → "무엇을 찾고 있나?"
    K: Key    → "각 단어가 어떤 내용인가?"
    V: Value  → "실제로 가져올 정보"
    '''

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        '''
        Args:
            Q: (batch, heads, seq_len, d_k)
            K: (batch, heads, seq_len, d_k)
            V: (batch, heads, seq_len, d_v)
            mask: 패딩된 위치 무시용

        Returns:
            output: (batch, heads, seq_len, d_v)
            attn_weights: (batch, heads, seq_len, seq_len)
        '''

        d_k = Q.size(-1)
        # d_k: Query/Key 벡터의 차원 수

        # ① QK^T: 유사도 계산
        # Q: (batch, heads, seq_len, d_k)
        # K^T: (batch, heads, d_k, seq_len)
        # scores: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # 각 단어쌍의 유사도 행렬

        # ② / √d_k: 스케일 조정
        # d_k가 크면 내적값이 너무 커짐
        # → softmax 후 그래디언트 소실
        # → √d_k로 나눠서 안정화!
        scores = scores / math.sqrt(d_k)

        # ③ Mask 처리 (패딩 위치 무시)
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0, float('-inf')
                # 패딩 위치를 -∞로 → softmax 후 0이 됨
            )

        # ④ softmax: 확률로 변환
        attn_weights = F.softmax(scores, dim=-1)
        # 각 단어가 다른 단어들에 얼마나 집중하는지

        # ⑤ × V: 가중합
        output = torch.matmul(attn_weights, V)
        # 중요한 단어의 Value를 더 많이 가져옴

        return output, attn_weights


# ==============================================
# PART 2. Multi-Head Attention
# ==============================================
# 논문 수식:
# MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
# head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
# ==============================================

class MultiHeadAttention(nn.Module):
    '''
    논문 Section 3.2.2

    h개의 Attention을 병렬로 실행
    각각 다른 관점으로 관계 파악
    결과를 합쳐서 최종 출력

    비유: 여러 전문가가 각자 분석 후 합의
    '''

    def __init__(self, d_model, num_heads):
        '''
        Args:
            d_model:   전체 임베딩 차원 (논문: 512)
            num_heads: Attention 헤드 수 (논문: 8)
        '''
        super().__init__()

        assert d_model % num_heads == 0
        # d_model이 num_heads로 나눠져야 함!

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads
        # 각 헤드의 차원 = 512 / 8 = 64

        # 논문: W^Q, W^K, W^V, W^O
        # 각 헤드별로 따로 만들지 않고
        # 한 번에 만들어서 나누는 방식
        self.W_Q = nn.Linear(d_model, d_model)
        # d_model → d_model (내부적으로 h×d_k)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        # Concat 후 최종 변환

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        '''
        (batch, seq_len, d_model)
        → (batch, num_heads, seq_len, d_k)

        d_model을 num_heads개로 분할해서
        각 헤드가 독립적으로 Attention 계산
        '''
        batch_size, seq_len, _ = x.size()

        x = x.view(
            batch_size, seq_len,
            self.num_heads, self.d_k
        )
        # (batch, seq_len, heads, d_k)

        return x.transpose(1, 2)
        # (batch, heads, seq_len, d_k)

    def forward(self, Q, K, V, mask=None):
        '''
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: 패딩 마스크

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, heads, seq_len, seq_len)
        '''

        # ① 선형 변환
        # (batch, seq_len, d_model) → 각 헤드용으로 변환
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # ② 헤드 분할
        # (batch, seq_len, d_model)
        # → (batch, heads, seq_len, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # ③ 각 헤드에서 Attention 계산
        # (batch, heads, seq_len, d_k)
        attn_output, attn_weights = self.attention(
            Q, K, V, mask
        )

        # ④ 헤드 합치기
        batch_size, _, seq_len, _ = attn_output.size()

        attn_output = attn_output.transpose(1, 2)
        # (batch, seq_len, heads, d_k)

        attn_output = attn_output.contiguous().view(
            batch_size, seq_len, self.d_model
        )
        # (batch, seq_len, d_model)
        # Concat(head_1, ..., head_h)

        # ⑤ 최종 선형 변환 W^O
        output = self.W_O(attn_output)
        # (batch, seq_len, d_model)

        return output, attn_weights


# ==============================================
# PART 3. Positional Encoding
# ==============================================
# 논문 수식:
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# ==============================================

class PositionalEncoding(nn.Module):
    '''
    논문 Section 3.5

    Transformer는 순서를 모름
    → 위치 정보를 직접 더해줌!

    sin/cos를 쓰는 이유:
    ① 모든 위치가 고유한 값
    ② 상대적 위치 관계 표현 가능
    ③ 학습 데이터보다 긴 문장도 처리 가능
    '''

    def __init__(self, d_model, max_len=5000,
                 dropout=0.1):
        '''
        Args:
            d_model: 임베딩 차원
            max_len: 최대 문장 길이
            dropout: 드롭아웃 비율
        '''
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE 행렬 미리 계산
        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 위치 인덱스: (max_len, 1)
        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)

        # 분모 계산: 10000^(2i/d_model)
        # 로그 스케일로 계산 (수치 안정성)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # 짝수 인덱스: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) 형태로
        pe = pe.unsqueeze(0)

        # 학습되지 않는 상수로 등록
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
            임베딩 + 위치 정보
        '''
        x = x + self.pe[:, :x.size(1), :]
        # 입력 임베딩에 위치 정보 더함
        return self.dropout(x)


# ==============================================
# PART 4. Feed Forward Network
# ==============================================
# 논문 수식:
# FFN(x) = max(0, xW1 + b1)W2 + b2
# ==============================================

class FeedForward(nn.Module):
    '''
    논문 Section 3.3

    Attention 후 각 위치에 독립적으로 적용
    2층 FC + ReLU

    d_model → d_ff → d_model
    논문: d_model=512, d_ff=2048 (4배)
    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        # 512 → 2048 (확장)
        self.linear2 = nn.Linear(d_ff, d_model)
        # 2048 → 512 (축소)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1+b1)W2+b2
        x = F.relu(self.linear1(x))
        # 확장 + ReLU
        x = self.dropout(x)
        x = self.linear2(x)
        # 축소
        return x


# ==============================================
# PART 5. Encoder Block
# ==============================================
# 구조:
# x → Multi-Head Attention → Add&Norm
#   → Feed Forward        → Add&Norm
# ==============================================

class EncoderBlock(nn.Module):
    '''
    논문 Figure 1의 Encoder 한 층

    구조:
    ┌─────────────────────────────┐
    │  Multi-Head Self-Attention  │
    │  + Add & Norm               │
    │  Feed Forward               │
    │  + Add & Norm               │
    └─────────────────────────────┘

    Add & Norm = 잔차 연결 + LayerNorm
    ResNet의 Skip Connection과 같은 원리!
    '''

    def __init__(self, d_model, num_heads,
                 d_ff, dropout=0.1):
        super().__init__()

        # Multi-Head Self-Attention
        self.attention = MultiHeadAttention(
            d_model, num_heads
        )

        # Feed Forward
        self.feed_forward = FeedForward(
            d_model, d_ff, dropout
        )

        # Layer Normalization (2개)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # BN과 달리 각 샘플 내에서 정규화

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        '''
        Args:
            x: (batch, seq_len, d_model)
            mask: 패딩 마스크

        Returns:
            (batch, seq_len, d_model)
        '''

        # ① Multi-Head Self-Attention + Add&Norm
        # Self-Attention: Q=K=V=x (자기 자신을 봄)
        attn_output, attn_weights = self.attention(
            x, x, x, mask
            # Q, K, V 모두 x!
            # = 문장이 자기 자신의 모든 단어를 봄
        )
        x = self.norm1(
            x + self.dropout(attn_output)
            # Add (잔차 연결) & Norm
        )

        # ② Feed Forward + Add&Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(
            x + self.dropout(ff_output)
            # Add (잔차 연결) & Norm
        )

        return x, attn_weights


# ==============================================
# PART 6. Transformer Encoder 전체
# ==============================================

class TransformerEncoder(nn.Module):
    '''
    논문 Figure 1의 Encoder 전체

    구조:
    Embedding → Positional Encoding
    → EncoderBlock × N
    → 출력
    '''

    def __init__(self, vocab_size, d_model,
                 num_heads, d_ff, num_layers,
                 max_len=512, dropout=0.1):
        '''
        Args:
            vocab_size:  단어 사전 크기
            d_model:     임베딩 차원 (논문: 512)
            num_heads:   Attention 헤드 수 (논문: 8)
            d_ff:        FFN 중간 차원 (논문: 2048)
            num_layers:  Encoder 블록 수 (논문: 6)
            max_len:     최대 문장 길이
            dropout:     드롭아웃 비율 (논문: 0.1)
        '''
        super().__init__()

        # 단어 임베딩
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )
        # 각 단어를 d_model 차원 벡터로 변환

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(
            d_model, max_len, dropout
        )

        # N개의 Encoder Block
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads,
                        d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.d_model = d_model

    def forward(self, x, mask=None):
        '''
        Args:
            x: (batch, seq_len) 단어 인덱스
            mask: 패딩 마스크

        Returns:
            output: (batch, seq_len, d_model)
            all_attn_weights: 각 레이어의 Attention
        '''

        # ① 임베딩 + 스케일 조정
        # 논문: 임베딩에 √d_model 곱함
        x = self.embedding(x) * math.sqrt(self.d_model)
        # (batch, seq_len, d_model)

        # ② Positional Encoding 추가
        x = self.pos_encoding(x)
        # (batch, seq_len, d_model)

        # ③ N개의 Encoder Block 통과
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


# ==============================================
# PART 7. 감정 분류 모델
# (Transformer Encoder + Classifier)
# ==============================================

class SentimentClassifier(nn.Module):
    '''
    Transformer Encoder로 감정 분류
    긍정(1) / 부정(0) 분류

    구조:
    문장 → Transformer Encoder
         → [CLS] 토큰 출력
         → FC → 긍정/부정
    '''

    def __init__(self, vocab_size, d_model=128,
                 num_heads=4, d_ff=256,
                 num_layers=2, num_classes=2,
                 max_len=512, dropout=0.1):
        # 실습용으로 논문보다 작은 크기 사용
        # 논문: d_model=512, heads=8, layers=6
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads,
            d_ff, num_layers, max_len, dropout
        )

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        '''
        Args:
            x: (batch, seq_len) 단어 인덱스

        Returns:
            logits: (batch, num_classes)
        '''
        # Encoder 통과
        encoded, attn_weights = self.encoder(x, mask)
        # (batch, seq_len, d_model)

        # 첫 번째 토큰([CLS])의 출력으로 분류
        # BERT와 동일한 방식!
        cls_output = encoded[:, 0, :]
        # (batch, d_model)

        # 분류
        logits = self.classifier(cls_output)
        # (batch, 2)

        return logits, attn_weights


# ==============================================
# PART 8. 간단한 토크나이저 & 데이터셋
# ==============================================

class SimpleTokenizer:
    '''
    단어를 인덱스로 변환하는 간단한 토크나이저
    실제로는 BPE, WordPiece 등 사용
    '''

    def __init__(self):
        # 기본 토큰
        self.word2idx = {
            '<PAD>': 0,  # 패딩
            '<CLS>': 1,  # 문장 시작
            '<UNK>': 2,  # 미등록 단어
        }
        self.idx2word = {
            v: k for k, v in self.word2idx.items()
        }

    def build_vocab(self, sentences):
        '''문장들로부터 단어 사전 구축'''
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, sentence, max_len=20):
        '''문장 → 인덱스 리스트'''
        tokens = ['<CLS>'] + sentence.lower().split()
        # CLS 토큰을 맨 앞에 추가!

        indices = [
            self.word2idx.get(t,
                self.word2idx['<UNK>'])
            for t in tokens
        ]

        # 패딩 또는 자르기
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
            # 부족하면 PAD(0)로 채움
        else:
            indices = indices[:max_len]
            # 길면 자름

        return indices

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ==============================================
# PART 9. 학습 데이터 준비
# ==============================================

# 간단한 감정 분류 데이터
train_data = [
    # 긍정 (label=1)
    ("this movie is great and amazing", 1),
    ("i love this film very much", 1),
    ("wonderful performance by all actors", 1),
    ("absolutely fantastic and brilliant", 1),
    ("best movie i have ever seen", 1),
    ("incredible story with great acting", 1),
    ("highly recommend this wonderful film", 1),
    ("outstanding and beautiful movie", 1),
    ("perfect film with amazing story", 1),
    ("superb acting and great direction", 1),
    ("this is a masterpiece of cinema", 1),
    ("loved every moment of this film", 1),
    ("brilliant and touching story", 1),
    ("excellent movie highly recommended", 1),
    ("fantastic performance and great plot", 1),

    # 부정 (label=0)
    ("this movie is terrible and boring", 0),
    ("i hate this film so much", 0),
    ("worst movie i have ever seen", 0),
    ("absolutely awful and disappointing", 0),
    ("bad acting and poor story", 0),
    ("complete waste of time and money", 0),
    ("terrible direction and bad script", 0),
    ("disappointing and boring film", 0),
    ("awful movie with bad acting", 0),
    ("horrible story and poor direction", 0),
    ("dreadful film completely unwatchable", 0),
    ("terrible waste of time", 0),
    ("boring and completely pointless", 0),
    ("worst film i have ever watched", 0),
    ("poor acting and terrible plot", 0),
]

test_data = [
    ("amazing and wonderful experience", 1),
    ("great acting and brilliant story", 1),
    ("boring and terrible movie", 0),
    ("awful and disappointing film", 0),
    ("fantastic and incredible film", 1),
    ("horrible and bad acting", 0),
]

# 토크나이저 구축
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(
    [s for s, _ in train_data + test_data]
)

MAX_LEN = 20

# 데이터 텐서 변환
def prepare_data(data, tokenizer, max_len):
    inputs = torch.tensor([
        tokenizer.encode(s, max_len)
        for s, _ in data
    ])
    labels = torch.tensor([l for _, l in data])
    return inputs, labels

train_inputs, train_labels = prepare_data(
    train_data, tokenizer, MAX_LEN
)
test_inputs, test_labels = prepare_data(
    test_data, tokenizer, MAX_LEN
)

# 패딩 마스크 생성
def make_mask(inputs):
    # PAD(0)인 위치를 0으로 마스킹
    return (inputs != 0).unsqueeze(1).unsqueeze(2)
    # (batch, 1, 1, seq_len)

train_mask = make_mask(train_inputs)
test_mask  = make_mask(test_inputs)

print("=" * 55)
print("PART 1. 데이터 준비")
print("=" * 55)
print(f"훈련 데이터: {len(train_data)}개")
print(f"테스트 데이터: {len(test_data)}개")
print(f"단어 사전 크기: {tokenizer.vocab_size}개")
print(f"최대 문장 길이: {MAX_LEN}")
print()


# ==============================================
# PART 10. 모델 구조 확인
# ==============================================

print("=" * 55)
print("PART 2. 모델 구조 확인")
print("=" * 55)

model = SentimentClassifier(
    vocab_size=tokenizer.vocab_size,
    d_model=128,      # 논문: 512 (실습용 축소)
    num_heads=4,      # 논문: 8  (실습용 축소)
    d_ff=256,         # 논문: 2048 (실습용 축소)
    num_layers=2,     # 논문: 6  (실습용 축소)
    num_classes=2,
    max_len=MAX_LEN,
    dropout=0.1
).to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)
print(f"전체 파라미터: {total_params:,}개")
print()

# Forward Pass 테스트
test_input_sample = train_inputs[:4].to(device)
test_mask_sample  = train_mask[:4].to(device)

with torch.no_grad():
    output, attn = model(
        test_input_sample, test_mask_sample
    )

print(f"Forward Pass 테스트:")
print(f"입력 shape:  {test_input_sample.shape}")
print(f"출력 shape:  {output.shape}")
print(f"Attention shape: {attn[0].shape}")
# (batch, heads, seq_len, seq_len)
print()


# ==============================================
# PART 11. 학습
# ==============================================

print("=" * 55)
print("PART 3. 모델 학습")
print("=" * 55)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-9
    # 논문과 동일한 Adam 하이퍼파라미터!
)
criterion = nn.CrossEntropyLoss()

EPOCHS = 100

print(f"{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>9} | {'Test Acc':>8}")
print("-" * 42)

for epoch in range(1, EPOCHS + 1):
    # 학습
    model.train()
    inputs = train_inputs.to(device)
    labels = train_labels.to(device)
    masks  = train_mask.to(device)

    logits, _ = model(inputs, masks)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()

    # Gradient Clipping
    # 논문: 그래디언트가 너무 커지는 것 방지
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    optimizer.step()

    # 평가
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            # 훈련 정확도
            train_pred = logits.argmax(dim=1)
            train_acc  = (
                train_pred == labels
            ).float().mean() * 100

            # 테스트 정확도
            test_inp = test_inputs.to(device)
            test_lbl = test_labels.to(device)
            test_msk = test_mask.to(device)

            test_logits, _ = model(test_inp, test_msk)
            test_pred  = test_logits.argmax(dim=1)
            test_acc   = (
                test_pred == test_lbl
            ).float().mean() * 100

        print(f"{epoch:>6} | "
              f"{loss.item():>8.4f} | "
              f"{train_acc.item():>8.2f}% | "
              f"{test_acc.item():>7.2f}%")

print()


# ==============================================
# PART 12. Attention 시각화
# ==============================================
# Transformer의 핵심!
# 어떤 단어가 어떤 단어를 보는지 확인

print("=" * 55)
print("PART 4. Attention 가중치 시각화")
print("=" * 55)

model.eval()
sample_sentence = "this movie is great and amazing"
sample_input = torch.tensor(
    [tokenizer.encode(sample_sentence, MAX_LEN)]
).to(device)
sample_mask = make_mask(sample_input).to(device)

with torch.no_grad():
    _, attn_weights = model(sample_input, sample_mask)

# 첫 번째 레이어, 첫 번째 헤드의 Attention
attn = attn_weights[0][0, 0].cpu().numpy()
# (seq_len, seq_len)

tokens = ['<CLS>'] + sample_sentence.split()
tokens = tokens[:8]
# 보기 쉽게 앞 8개만

print(f"문장: '{sample_sentence}'")
print(f"\nAttention 가중치 (레이어1, 헤드1):")
print(f"행 = 이 단어가, 열 = 이 단어를 얼마나 보는지\n")

# 헤더 출력
print(f"{'':>8}", end="")
for t in tokens:
    print(f"{t:>8}", end="")
print()
print("-" * (8 + 8 * len(tokens)))

# Attention 행렬 출력
for i, token in enumerate(tokens):
    print(f"{token:>8}", end="")
    for j in range(len(tokens)):
        val = attn[i][j]
        print(f"{val:>8.3f}", end="")
    print()

print()


# ==============================================
# PART 13. 최종 예측 테스트
# ==============================================

print("=" * 55)
print("PART 5. 최종 예측 테스트")
print("=" * 55)

test_sentences = [
    ("this movie is absolutely wonderful", 1),
    ("terrible and boring film i hated", 0),
    ("great acting and amazing story", 1),
    ("worst movie ever so bad", 0),
    ("brilliant and fantastic performance", 1),
]

label_map = {0: "부정 😞", 1: "긍정 😊"}

print(f"{'문장':<40} | {'정답':^8} | {'예측':^8} | {'확신도':^8}")
print("-" * 75)

model.eval()
for sentence, true_label in test_sentences:
    inp  = torch.tensor(
        [tokenizer.encode(sentence, MAX_LEN)]
    ).to(device)
    msk  = make_mask(inp).to(device)

    with torch.no_grad():
        logits, _ = model(inp, msk)
        probs = F.softmax(logits, dim=1)
        pred  = logits.argmax(dim=1).item()
        conf  = probs[0][pred].item() * 100

    correct = "✅" if pred == true_label else "❌"
    print(f"{sentence:<40} | "
          f"{label_map[true_label]:^8} | "
          f"{label_map[pred]:^8} | "
          f"{conf:>6.1f}%  {correct}")

print()
print("Transformer 구현(Replication) 완료! 🎉")
print()
print("논문 핵심 구조 구현 완료:")
print("  ✅ Scaled Dot-Product Attention")
print("  ✅ Multi-Head Attention")
print("  ✅ Positional Encoding")
print("  ✅ Feed Forward Network")
print("  ✅ Add & Layer Norm")
print("  ✅ Encoder Block")
print("  ✅ Transformer Encoder")