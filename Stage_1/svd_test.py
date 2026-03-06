import numpy as np
import matplotlib.pyplot as plt

# 흑백 이미지를 SVD로 압축하는 예재
# 이미지를 행렬로 보고, 상위 k개의 특이값만 유지

def svd_compress(image_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    SVD를 이용한 이미지 압축
    
    수식: A ≈ U_k * Σ_k * V_k^T
    - U_k: 상위 k개 왼쪽 특이벡터
    - Σ_k: 상위 k개 특이값
    - V_k^T: 상위 k개 오른쪽 특이벡터
    
    Args:
        image_matrix: 2D numpy array (H, W) - 흑백 이미지
        k: 유지할 특이값 개수 (클수록 원본에 가까움)
    
    Returns:
        압축된 이미지 행렬
    """
    # SVD 수행: image = U @ np.diag(S) @ Vt
    # U.shape = (H, H), S.shape = (min(H,W),), Vt.shape = (W, W)
    # 이미지를 세 개로 분해 한다는 뜻
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=True)
    # U  : (64, 64)  → 이미지의 "방향 정보"
		# S  : (64,)     → 중요도 순서로 정렬된 숫자 64개
		# Vt : (64, 64)  → 이미지의 "구조 정보"
    
    # 상위 k개만 사용 (나머지는 버림 → 압축)
    # 수식: A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    U_k  = U[:, :k]    # U에서 앞 k개 열만
    S_k  = np.diag(S[:k])  # S에서 앞 k개만 (나머지 버림)
    Vt_k = Vt[:k, :]   # Vt에서 앞 k개 행만
    
    '''
    k=1 이면:
        중요도 1위짜리만 남기고 나머지 전부 버림
        → 이미지가 뭉개지지만 대략적인 형태는 유지
        
    k=64 이면:
        전부 다 남김
        → 원본과 완전히 같음
    '''
    
    
    # 재구성 : 세 조각을 다시 합쳐서 이미지 복원
    reconstructed = U_k @ S_k @ Vt_k  # (H, W)
    

    # 픽셀 범위 클리핑 [0, 255]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)
    
# 테스트 (실제 이미지 없이 랜덤 행렬로 시연)
np.random.seed(42)
fake_image = np.random.randint(0, 256, (64, 64)).astype(np.float64)

for k in [1, 5, 10, 30, 64]:
		compressed = svd_compress(fake_image, k)
		# 압축률: k*(H+W+1) / (H*W) 개의 숫자만 저장
		ratio = k * (64 + 64 + 1) / (64 * 64)
		print(f"k={k:3d} | 압축률: {ratio:.2%} | 원본 대비 사용 숫자: {k*(64+64+1) / (64*64):.4f}")


fig, axes = plt.subplots(1, 5, figsize=(15, 3))
k_values = [1, 5, 10, 30, 64]

for ax, k in zip(axes, k_values):
    compressed = svd_compress(fake_image, k)
    ax.imshow(compressed, cmap='gray')
    ax.set_title(f'k={k}\n({k*129/4096:.1%})')
    ax.axis('off')

plt.suptitle('SVD 압축: k가 클수록 원본에 가까워짐')
plt.tight_layout()
plt.show()

'''
실행 결과 해석
k=  1 | 압축률: 3.15% | 원본 대비 사용 숫자: 0.0315 → 전체 정보의 3%만 사용
k=  5 | 압축률: 15.75% | 원본 대비 사용 숫자: 0.1575 → 전체 정보의 15%만 사용
k= 10 | 압축률: 31.49% | 원본 대비 사용 숫자: 0.3149 → 전체 정보의 31%만 사용
k= 30 | 압축률: 94.48% | 원본 대비 사용 숫자: 0.9448 → 전체 정보의 94%만 사용
k= 64 | 압축률: 201.56% | 원본 대비 사용 숫자: 2.0156 → 원본보다 2배 더 많은 숫자 필요!
'''