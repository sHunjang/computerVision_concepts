import numpy as np
import cv2
from typing import Tuple, List

class ImageProcessingPipeline:
    """
    Stage 1 종합 실습: 고전 이미지 처리 파이프라인
    
    이 클래스는 다음을 통합한다:
    - 이미지 로딩 및 텐서 변환
    - 다양한 필터 적용 (Convolution)
    - 에지 검출 (Sobel, Canny)
    - 히스토그램 분석
    
    핵심 학습: 딥러닝 이전에 존재했던 "손으로 설계한 특징"들이
              CNN에서는 학습을 통해 자동으로 발견된다.
    """
    
    def __init__(self):
        # Sobel 필터 (에지 검출)
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float32)
        
        # Gaussian 블러 커널 (노이즈 제거)
        self.gaussian_3x3 = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0  # 가중치 합 = 1 (정규화)
        
        # Laplacian (에지 + 방향성)
        self.laplacian = np.array([
            [ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]
        ], dtype=np.float32)
    
    def load_and_preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지 로딩 및 전처리
        
        Returns:
            (bgr_image, gray_image)
        """
        # 이미지 로딩
        bgr = cv2.imread(image_path)
        if bgr is None:
            # 파일이 없으면 테스트용 이미지 생성
            bgr = self._create_test_image()
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr, gray
    
    def _create_test_image(self) -> np.ndarray:
        """테스트용 합성 이미지 생성"""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        # 사각형
        cv2.rectangle(img, (20, 20), (60, 60), (255, 100, 50), -1)
        # 원
        cv2.circle(img, (90, 70), 25, (50, 200, 100), -1)
        # 선
        cv2.line(img, (0, 100), (128, 100), (200, 50, 200), 2)
        return img
    
    def apply_conv2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        padding: int = 1
    ) -> np.ndarray:
        """수식 기반 Convolution 적용"""
        H, W = image.shape[:2]
        Kh, Kw = kernel.shape
        
        H_out = (H - Kh + 2 * padding) + 1
        W_out = (W - Kw + 2 * padding) + 1
        
        padded = np.pad(image.astype(np.float32), padding, mode='constant')
        output = np.zeros((H_out, W_out), dtype=np.float32)
        
        for i in range(H_out):
            for j in range(W_out):
                patch = padded[i:i+Kh, j:j+Kw]
                output[i, j] = np.sum(patch * kernel)
        
        return output
    
    def detect_edges(self, gray: np.ndarray) -> dict:
        """
        에지 검출 파이프라인
        
        Sobel 에지 크기:
        |G| = √(Gx^2 + Gy^2)
        
        방향:
        θ = arctan(Gy / Gx)
        """
        # 1. 가우시안 블러로 노이즈 제거
        # → Sobel 필터로 경계선 찾기
        # → 에지 크기 = √(Gx² + Gy²)
        # (실제 에지 검출 전 필수 단계)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Sobel 그래디언트 계산
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)  # x 방향
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)  # y 방향
        
        # 3. 에지 크기: |G| = √(Gx^2 + Gy^2)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # 4. 에지 방향: θ = arctan2(Gy, Gx) [degree]
        direction = np.arctan2(gy, gx) * (180 / np.pi)
        
        # 5. Canny 에지 (OpenCV 사용)
        canny = cv2.Canny(blurred, threshold1=50, threshold2=150)
        
        return {
            'gradient_x':  gx,
            'gradient_y':  gy,
            'magnitude':   magnitude,
            'direction':   direction,
            'canny':       canny
        }
    
    def compute_histogram(self, gray: np.ndarray) -> np.ndarray:
        """
        이미지 히스토그램 계산
        
        히스토그램: 각 픽셀값(0~255)이 몇 번 등장하는지
        → 이미지의 "밝기 분포"를 나타냄
        → 딥러닝에서 Batch Normalization의 필요성과 연결됨
        """
        hist = np.zeros(256, dtype=np.int32)
        
        # 각 픽셀값의 빈도 계산
        for pixel_val in gray.flatten():
            hist[pixel_val] += 1
        
        return hist
    
    def normalize_to_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        이미지를 딥러닝 입력 텐서로 변환
        
        변환 과정:
        (H, W, C) BGR uint8 [0,255]
        → (H, W, C) RGB float32 [0,1]
        → (C, H, W) float32  [채널 우선]
        → (1, C, H, W) float32 [배치 추가]
        
        정규화 수식: x_norm = (x - μ) / σ
        ImageNet 표준: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]
        """
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # uint8 → float32, [0,255] → [0,1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet 평균/표준편차로 정규화
        # 수식: x_out = (x - μ) / σ (채널별)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # HWC → CHW
        chw = np.transpose(normalized, (2, 0, 1))
        
        # 배치 차원 추가: CHW → NCHW
        nchw = np.expand_dims(chw, axis=0)
        
        return nchw
    
    def run_demo(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("Stage 1 종합 실습: 이미지 처리 파이프라인")
        print("=" * 60)
        
        # 이미지 생성
        bgr_img, gray_img = self.load_and_preprocess("test.jpg")
        print(f"\n[1] 이미지 로딩")
        print(f"    BGR 텐서 shape: {bgr_img.shape}  (H, W, C)")
        print(f"    Gray 텐서 shape: {gray_img.shape}  (H, W)")
        
        # 에지 검출
        edges = self.detect_edges(gray_img)
        print(f"\n[2] 에지 검출")
        print(f"    그래디언트 크기 최댓값: {edges['magnitude'].max():.1f}")
        print(f"    Canny 에지 픽셀 수: {edges['canny'].sum() // 255}")
        
        # 히스토그램
        hist = self.compute_histogram(gray_img)
        print(f"\n[3] 히스토그램")
        print(f"    가장 많은 픽셀값: {hist.argmax()} (빈도: {hist.max()})")
        print(f"    평균 밝기: {np.sum(np.arange(256) * hist) / hist.sum():.1f}")
        
        # 딥러닝 텐서 변환
        tensor = self.normalize_to_tensor(bgr_img)
        print(f"\n[4] 딥러닝 텐서 변환")
        print(f"    입력:  {bgr_img.shape} (H,W,C) uint8 [0,255]")
        print(f"    출력:  {tensor.shape} (N,C,H,W) float32 [정규화됨]")
        print(f"    채널 0 범위: [{tensor[0,0].min():.2f}, {tensor[0,0].max():.2f}]")
        
        print(f"\n[학습 포인트]")
        print(f"  → Sobel 필터 = 수직/수평 에지를 감지하는 '손으로 설계한' 커널")
        print(f"  → CNN에서는 이런 필터를 데이터로부터 '자동 학습'한다")
        print(f"  → 이것이 딥러닝의 핵심 아이디어!")


# 파이프라인 실행
pipeline = ImageProcessingPipeline()
pipeline.run_demo()