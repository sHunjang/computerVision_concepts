import numpy as np
import cv2

def explore_image_representation():
    """
    이미지의 수학적 표현 탐구
    
    핵심 개념:
    - 이미지 = 픽셀값의 행렬 = 숫자의 배열
    - RGB 이미지 = (H, W, 3) 텐서
    - 딥러닝 입력 = (N, C, H, W) 텐서 (PyTorch 표준)
    """
    
    # 1. 가상 이미지 생성 (실제로는 cv2.imread 사용)
    H, W = 4, 4  # 4x4 픽셀 이미지
    
    # 흑백 이미지 만들기: 2D 행렬
    gray_image = np.array([    # 숫자 표 하나
		[ 10, 20, 30, 40],
        [ 50,  60,  70,  80],
        [ 90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.uint8)
    
    print("=== 흑백 이미지 ===")
    print(f"Shape: {gray_image.shape}  → (H={H}, W={W})")
    print(f"dtype: {gray_image.dtype}  → 0~255 정수")
    print(f"이미지 행렬:\n{gray_image}\n")
    
    # 2. RGB 이미지 만들기: 3D 텐서 (H, W, C)
    # 채널 순서: [빨강, 초록, 파랑]
    # 표 3장
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = gray_image       # R 채널
    rgb_image[:, :, 1] = gray_image       # G 채널
    rgb_image[:, :, 2] = 255 - gray_image # B 채널

    print("=== RGB 이미지 ===")
    print(f"Shape: {rgb_image.shape}  → (H={H}, W={W}, C=3)")
    print(f"픽셀 (0,0)의 RGB 값: {rgb_image[0, 0]}  → [R={rgb_image[0,0,0]}, G={rgb_image[0,0,1]}, B={rgb_image[0,0,2]}]")

    # 3. 딥러닝 입력으로 변환: (H, W, C) → (N, C, H, W)
    # 정규화: [0, 255] → [0.0, 1.0] 줄이기
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # (H, W, C) → (C, H, W) (채널을 앞으로)
    chw = np.transpose(normalized, (2, 0, 1))  # (3, 4, 4)
    
    # 배치 차원 추가 → (1, C, H, W)
    nchw = np.expand_dims(chw, axis=0)  # (1, 3, 4, 4)
    
    print(f"\n=== 딥러닝 텐서 변환 ===")
    print(f"원본 (H,W,C): {rgb_image.shape}")
    print(f"정규화 후:    {normalized.shape}")
    print(f"CHW 변환:     {chw.shape}")
    print(f"배치 추가:    {nchw.shape}  → (N=1, C=3, H=4, W=4)")
    print(f"\nR 채널 행렬:\n{nchw[0, 0].round(2)}")
    
    # 4. 픽셀 접근법 비교
    print(f"\n=== 픽셀 접근 ===")
    h, w = 1, 2  # 2행 3열 픽셀
    print(f"HWC 형식 [h={h}, w={w}, :]: {rgb_image[h, w, :]}")
    print(f"CHW 형식 [:, h={h}, w={w}]: {chw[:, h, w].round(2)}")

explore_image_representation()