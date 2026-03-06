import numpy as np
import cv2
from typing import Tuple

def conv2d_from_scratch(
		image: np.ndarray,
		kernel: np.ndarray,
		padding: int = 0,
		stride: int = 1
) -> np.ndarray:
    """
    2D Convolution을 NumPy로 처음부터 구현
    
    수식: O[i,j] = Σ_m Σ_n I[i*s+m, j*s+n] * K[m,n]
    
    Args:
        image:   입력 이미지 (H, W) - 흑백
        kernel:  합성곱 필터 (Kh, Kw)
        padding: 테두리 zero-padding 크기
        stride:  필터 이동 간격
    
    Returns:
        Feature Map (H_out, W_out)
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    
    # 출력 크기 계산
    # H_out = floor((H - Kh + 2*P) / S) + 1
    H_out = (H - Kh + 2 * padding) // stride + 1
    W_out = (W - Kw + 2 * padding) // stride + 1
    
    # Padding 적용: 테두리에 0 추가
    if padding > 0:
        image_padded = np.pad(
            image,
            pad_width=padding,
            mode='constant',
            constant_values=0
        )
    else:
        image_padded = image
	  
	# 출력 행렬 초기화
    output = np.zeros((H_out, W_out), dtype=np.float32)
	  
	# Convolution 핵심 루프 - 이미지 위에서 필터 슬라이딩
	# i, j: 출력 Feature Map의 위치
    for i in range(H_out):
        for j in range(W_out):
            # 현재 위치에서 커널 크기만큼 이미지 패치 추출
            # stride를 고려한 입력 좌표: (i*stride, j*stride)
            i_start = i * stride
            j_start = j * stride
					  
            patch = image_padded[   # 3x3 조각 추출
                i_start : i_start + Kh,
                j_start : j_start + Kw
                ]   # (Kh, Kw)
					  
            # 내적 (element-wise 곱의 합) = convolution의 본질
            # 수식: 0[i,j] = Σ_m Σ_n patch[m,n] * K[m,n]
            output[i, j] = np.sum(patch * kernel)
	
    return output

def demonstrate_filters():
    """
    다양한 필터 효과 시연
    """
    # 간단한 테스트 이미지 (실제로는 cv2.imread 사용)
    # 수직선이 있는 이미지
    image = np.zeros((8, 8), dtype=np.float32)
    image[:, 3:5] = 200  # 수직선
    image[3:5, :] = 150  # 수평선
    
    print("=== 입력 이미지 (8x8) ===")
    print(image.astype(int))
    print()
    
    # 1. 수직 에지 검출 필터 (Sobel X) - 필터 정의
    sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
    ], dtype=np.float32)
    
    # 2. 수평 에지 검출 필터 (Sobel Y) - 필터 정의
    sobel_y = np.array([
            [-1, -2, 1],
            [0, 0, 0],
            [1, 2, 1]
    ], dtype=np.float32)
    
    # 3. Blur 필터 (평균)
    blur = np.ones((3, 3), dtype=np.float32) / 9.0
    
    # Convolution 적용
    edge_x = conv2d_from_scratch(image, sobel_x, padding=1)
    edge_y = conv2d_from_scratch(image, sobel_y, padding=1)
    blurred = conv2d_from_scratch(image, blur, padding=1)
    
    print("=== Sobel X (수직 에지) ===")
    print(edge_x.astype(int))
    print("→ 수직선 경계(열 3, 5)에서 큰 값 발생\n")

    print("=== Sobel Y (수평 에지) ===")
    print(edge_y.astype(int))
    print("→ 수평선 경계(행 3, 5)에서 큰 값 발생\n")

    # 출력 크기 검증
    print("=== 출력 크기 확인 ===")
    print(f"입력: (8, 8), 커널: (3, 3), P=1, S=1")
    H_out = (8 - 3 + 2*1) // 1 + 1
    print(f"계산: H_out = (8 - 3 + 2×1) / 1 + 1 = {H_out}")
    print(f"실제 출력 크기: {edge_x.shape}")
    
demonstrate_filters()