# AI_picfilter 기술 아키텍처 문서

## 1. 시스템 개요

AI_picfilter는 레퍼런스 이미지의 시각적 특성(색상, 조명, 질감, 분위기)을 학습하여 재사용 가능한 필터를 생성하는 시스템이다. 최종 목표는 사용자가 원하는 스타일의 사진 한 장(혹은 소수)을 입력하면, 해당 스타일을 다른 이미지에 적용할 수 있는 경량 필터를 자동으로 생성하는 것이다.

---

## 2. AI 모델 방법론

### 2.1 Neural Style Transfer (Deep Photo Style Transfer)

**핵심 원리**: 사전 학습된 VGG-19 CNN의 계층별 feature를 활용하여 content(구조)와 style(색감/텍스처)을 분리한다. Gram Matrix로 스타일 통계를 수치화하고, photorealism regularization(로컬 아핀 제약)을 적용하여 사진적 사실감을 보존한다.

**아키텍처 구성**:
- Content Loss: VGG-19 중간 계층(conv4_2)의 feature 일치
- Style Loss: 다중 계층의 Gram Matrix 일치
- Photorealism Regularization: 라플라시안 행렬 기반 로컬 아핀 제약
- Semantic Segmentation: 영역별 독립적 스타일 전이를 위한 마스크 입력

**적용 시나리오**: 고품질 단건 스타일 전이. 참조 이미지 1장으로부터 정교한 색감/분위기를 복제해야 할 때 적합하다.

**한계**: 최적화 기반(iterative)이므로 추론이 느리고, CUDA + Torch/Lua + MATLAB 종속성이 높다.

---

### 2.2 Image-Adaptive 3D LUTs

**핵심 원리**: 경량 CNN이 입력 이미지의 저해상도 버전을 분석하여 이미지의 조명/분위기를 파악한 뒤, 미리 학습된 N개의 basis LUT를 선형 결합하는 가중치를 예측한다.

**아키텍처 구성**:
- Backbone CNN: 입력 이미지 다운샘플(256x256) -> 가중치 벡터 예측
- Basis LUTs: 학습 가능한 N개의 3D LUT (일반적으로 N=3~5)
- Trilinear Interpolation: 예측된 가중치로 블렌딩한 최종 LUT를 원본 해상도에 적용
- Smoothness Regularization: LUT의 단조성(monotonicity) 보장

**적용 시나리오**: 실시간 필터 적용이 필요한 환경. 4K 해상도에서 2ms 미만으로 처리 가능하며, 480p에서 학습한 모델이 4K에 그대로 적용된다.

**학습 방식**: Paired(보정 전/후 이미지 쌍) 또는 Unpaired 데이터 모두 지원.

---

### 2.3 Neural Implicit LUTs (NILUT)

**핵심 원리**: 색상 변환을 MLP(다층 퍼셉트론)의 연속 함수로 학습한다. 입력 RGB 좌표를 받아 변환된 RGB를 직접 출력하므로, 이산적 LUT 그리드의 보간 오차가 없다.

**아키텍처 구성**:
- Input: RGB 3채널 + 선택적 스타일 조건 벡터
- Network: 소규모 MLP (3~5 hidden layers, 각 64~256 units)
- Output: 변환된 RGB 3채널
- Multi-style: 스타일 인덱스를 조건 입력으로 넣어 단일 모델에서 512개 이상의 스타일을 지원
- Blending: 스타일 간 연속적 보간(interpolation)이 가능

**적용 시나리오**: 다수의 스타일을 하나의 경량 모델로 통합 관리해야 할 때. 스타일 간 블렌딩이 필요한 인터랙티브 UI에 적합하다.

**특징**: 0.25MB 미만으로 512개 스타일을 포함할 수 있어 모바일/엣지 배포에 최적이다.

---

### 2.4 GAN 기반 스타일 전이 (StarGAN v2)

**핵심 원리**: 단일 Generator가 다중 도메인 간 이미지 변환을 수행한다. Style Encoder가 참조 이미지에서 스타일 코드를 추출하고, Mapping Network가 랜덤 스타일을 생성한다.

**아키텍처 구성**:
- Generator: ResNet 기반 인코더-디코더, AdaIN(Adaptive Instance Normalization) 적용
- Discriminator: 멀티스케일 판별기, 도메인별 출력 헤드
- Style Encoder: 참조 이미지 -> 스타일 벡터 추출
- Mapping Network: 랜덤 노이즈 -> 스타일 벡터 생성
- Loss: Adversarial + Style Reconstruction + Cycle Consistency + Style Diversification

**적용 시나리오**: 색감뿐 아니라 전반적인 분위기(조명, 텍스처, 피부톤 등)를 복합적으로 전이해야 할 때. 특히 인물/동물 사진 등 도메인 특화 작업에 강하다.

**한계**: 학습 데이터가 충분히 필요하고, 모델 크기가 크며(수십~수백 MB), 추론 속도가 LUT 계열 대비 느리다.

---

### 2.5 Diffusion-based LUT (D-LUT)

**핵심 원리**: Score-matching(확산 과정)을 통해 참조 이미지의 색상 분포를 학습하고, 학습 결과를 표준 3D LUT(.cube 파일)로 추출한다.

**아키텍처 구성**:
- Forward Diffusion: 참조 이미지의 색상 분포에 점진적 노이즈 추가
- Reverse Process: 학습된 score function으로 노이즈 제거, 색상 매핑 복원
- LUT Extraction: 학습된 변환을 샘플링하여 표준 .cube 파일로 export
- Content Preservation: 확산 과정에서 구조적 정보를 조건으로 주입

**적용 시나리오**: 학습 결과를 Photoshop, Premiere Pro 등 상용 소프트웨어에서 즉시 사용해야 할 때. 연구와 실무 워크플로우를 연결하는 브릿지 역할.

**특징**: 고해상도에서도 아티팩트가 거의 없으며, 출력이 .cube 표준 포맷이므로 범용성이 높다.

---

## 3. 모델별 비교표

| 평가 항목 | Neural Style Transfer | Image-Adaptive 3D LUT | NILUT | StarGAN v2 | D-LUT |
|---|---|---|---|---|---|
| **색상 정확도** | 높음 (주관적 품질 우수) | 높음 (ΔE < 1.0 달성 가능) | 매우 높음 (연속 함수, ΔE < 1.0) | 중간 (GAN 특성상 변동) | 높음 (정교한 분포 학습) |
| **추론 속도** | 매우 느림 (수초~수분) | 매우 빠름 (<2ms @4K) | 빠름 (<16ms @4K) | 느림 (수십~수백ms) | 중간 (LUT 추출 후 빠름) |
| **모델 크기** | 큼 (VGG-19: ~550MB) | 작음 (수백 KB~수 MB) | 매우 작음 (<0.25MB) | 큼 (수십~수백 MB) | 중간 (학습 모델) + 작음 (.cube) |
| **구현 난이도** | 중간 (Torch/Lua 레거시) | 낮음 (PyTorch, 잘 정리된 코드) | 낮음 (단순 MLP 구조) | 높음 (복잡한 GAN 학습) | 높음 (확산 모델 이해 필요) |
| **데이터 요구량** | 최소 (참조 1장) | 중간 (이미지 쌍 수백~수천) | 낮음 (.cube 파일 or 이미지 쌍) | 높음 (도메인당 수천 장) | 중간 (색상 분포 학습용) |
| **구조 보존** | 우수 (아핀 제약) | 우수 (LUT는 픽셀 독립) | 우수 (픽셀 독립 변환) | 보통 (GAN 아티팩트 가능) | 우수 |
| **다중 스타일** | 불가 (1회 1스타일) | 제한적 (모델별 1스타일) | 우수 (단일 모델 512+) | 우수 (다중 도메인) | 제한적 (LUT별 1스타일) |
| **상용 SW 호환** | 불가 | 가능 (.cube 변환 필요) | 불가 (자체 추론 필요) | 불가 | 우수 (.cube 직접 출력) |
| **실시간 적용** | 불가 | 가능 | 가능 | 불가 | LUT 추출 후 가능 |

---

## 4. 품질 기준 달성 전략

### 4.1 색상 정확도: CIE76 ΔE < 1.0

- **1차 검증**: .cube 파일 또는 학습된 LUT를 ColorChecker 차트에 적용하여 ΔE 측정
- **학습 전략**: Perceptual loss와 함께 ΔE를 직접 loss에 포함하는 custom loss function 설계
- **데이터**: 전문 보정 전/후 이미지 쌍 (Adobe FiveK, MIT-Adobe 5K dataset 활용)
- **적합 모델**: NILUT, Image-Adaptive 3D LUT (두 모델 모두 ΔE < 1.0 달성 사례 보고)

### 4.2 추론 속도: 4K 기준 < 16ms

- **LUT 계열 모델 활용**: Image-Adaptive 3D LUT(<2ms), NILUT(<16ms)는 기본적으로 목표 충족
- **GPU 최적화**: TensorRT, ONNX Runtime으로 추론 가속
- **해상도 독립 학습**: 저해상도(480p)에서 학습 -> 고해상도(4K) 적용 (LUT의 해상도 불변 특성 활용)

### 4.3 모델 크기: < 10MB

- **NILUT**: 0.25MB 미만으로 목표 초과 달성
- **Image-Adaptive 3D LUT**: backbone CNN + basis LUT 합산 수 MB로 목표 충족
- **경량화 기법**: Knowledge distillation, weight pruning, INT8 quantization 적용 가능

### 4.4 구조 보존: PSNR / SSIM 최대화

- **평가 파이프라인**: 변환 전/후 이미지의 엣지 맵(Canny/Sobel) 비교, SSIM 채널별 분석
- **LUT 계열의 본질적 장점**: 픽셀 독립 색상 변환이므로 공간 구조에 영향을 주지 않음
- **GAN/NST 사용 시**: Content loss 가중치를 높여 구조 왜곡 억제

---

## 5. 전체 시스템 아키텍처

```
[User Input]
    |
    v
+-------------------+     +----------------------+
| Style Reference   | --> | Style Analyzer       |
| (1~N images)      |     | (Feature Extraction) |
+-------------------+     +----------+-----------+
                                     |
                                     v
                          +----------+-----------+
                          | Filter Generator     |
                          | (NILUT / 3D LUT /    |
                          |  D-LUT training)     |
                          +----------+-----------+
                                     |
                                     v
                          +----------+-----------+
                          | Filter Artifact      |
                          | (.cube / .pt model)  |
                          +----------+-----------+
                                     |
                                     v
+-------------------+     +----------+-----------+
| Target Image      | --> | Filter Applicator    | --> [Output Image]
|                   |     | (Real-time LUT apply)|
+-------------------+     +----------------------+
```

핵심 파이프라인:
1. **Style Analysis**: 레퍼런스 이미지에서 색상/조명/분위기 특성 추출
2. **Filter Generation**: 추출된 특성을 기반으로 LUT 또는 경량 모델 학습/생성
3. **Filter Application**: 생성된 필터를 대상 이미지에 실시간 적용
4. **Export**: .cube 파일 또는 경량 모델(.pt, .onnx)로 내보내기
