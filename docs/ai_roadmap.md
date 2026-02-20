# AI_picfilter 구현 로드맵

## 권장 시작점

**NILUT (Neural Implicit LUT)부터 시작할 것을 권장한다.**

근거:
1. **구현 난이도가 낮다**: 핵심 구조가 단순 MLP이므로 PyTorch 경험만 있으면 빠르게 프로토타입 가능
2. **데이터 요구량이 적다**: 기존 .cube 파일을 학습 데이터로 활용 가능 (fit.py 스크립트 제공)
3. **품질 기준을 즉시 충족한다**: ΔE < 1.0, 모델 크기 < 0.25MB, 실시간 추론 모두 기본 달성
4. **다중 스타일 지원**: 단일 모델로 수백 개 스타일을 관리할 수 있어 MVP에 적합
5. **확장성**: Phase 2의 Image-Adaptive 3D LUT와 자연스럽게 결합 가능 (LUT 계열의 공통 기반)

---

## Phase 1: 기초 파이프라인 구축 (4~6주)

### 목표
- 데이터 파이프라인, 학습/추론 루프, 평가 시스템의 기본 골격 확립
- NILUT 기반 첫 번째 동작하는 필터 생성 시스템 완성

### 세부 작업

**1.1 프로젝트 환경 세팅 (1주)**
- PyTorch 기반 개발 환경 구성 (Python 3.10+, PyTorch 2.x, CUDA 지원)
- 데이터 디렉토리 구조 및 설정 파일 체계 확립
- 로깅(WandB/TensorBoard), 실험 관리(Hydra/OmegaConf) 설정
- CI/테스트 프레임워크 구성

**1.2 데이터 파이프라인 (1~2주)**
- .cube 파일 파서 구현 (3D LUT 로딩/저장)
- Hald 이미지 기반 스타일 추출 파이프라인
- 이미지 쌍 데이터셋 로더 (Adobe FiveK 등)
- 데이터 전처리: 정규화, 색공간 변환(sRGB <-> Linear), 증강

**1.3 NILUT 모델 구현 (2주)**
- MLP 기반 NILUT 아키텍처 구현
  - Input: RGB (3) + style condition vector
  - Hidden: 3~5 layers, 64~256 units, ReLU/GELU
  - Output: RGB (3)
- 단일 스타일 학습 (fit.py 참고): .cube -> NILUT 변환
- 다중 스타일 학습: 스타일 인덱스를 조건 입력으로 추가
- 스타일 블렌딩: 두 스타일 간 연속 보간 구현

**1.4 평가 시스템 (1주)**
- CIE76 ΔE 계산 모듈
- PSNR, SSIM 계산 모듈
- 추론 시간 벤치마크 스크립트
- 시각적 비교 이미지 생성 (grid visualization)

### 검증 방법
- [ ] .cube 파일 -> NILUT 변환 후 ΔE < 1.0 확인
- [ ] 단일 이미지에 대해 스타일 적용 결과 시각적 검증
- [ ] 512개 스타일 동시 학습 후 모델 크기 < 0.25MB 확인
- [ ] 스타일 블렌딩(A 50% + B 50%)이 자연스러운지 시각적 확인
- [ ] 4K 이미지 추론 시간 < 16ms 확인

---

## Phase 2: 적응형 LUT 및 실시간 시스템 (4~6주)

### 목표
- 이미지 내용에 따라 적응적으로 필터를 적용하는 Image-Adaptive 3D LUT 구현
- .cube 파일 export로 상용 소프트웨어 호환성 확보
- 실시간 4K 처리 파이프라인 완성

### 세부 작업

**2.1 Image-Adaptive 3D LUT 구현 (2~3주)**
- Backbone CNN 설계: 경량 네트워크(MobileNet-v3 또는 커스텀)
- Basis LUT 학습: N개(3~5)의 학습 가능한 3D LUT 파라미터
- 가중치 예측: CNN -> softmax -> basis LUT 블렌딩
- Trilinear interpolation 모듈 (CUDA 커널 or PyTorch 구현)
- Smoothness/Monotonicity regularization loss

**2.2 학습 파이프라인 확장 (1~2주)**
- Paired 학습: L1/L2 loss + perceptual loss
- Unpaired 학습: Adversarial loss + cycle consistency (선택)
- 학습 데이터 확장: Adobe FiveK, PPR10K 등
- 학습률 스케줄링, early stopping, best model checkpointing

**2.3 Export 파이프라인 (1주)**
- 학습된 LUT를 .cube 파일로 변환하는 유틸리티
- ONNX / TorchScript export
- 모바일 배포용 TFLite 변환 (선택)

### 검증 방법
- [ ] Adobe FiveK 테스트셋에서 PSNR, SSIM, ΔE 정량 평가
- [ ] 4K 이미지 추론 < 2ms 확인 (GPU)
- [ ] 생성된 .cube 파일을 Photoshop/Premiere에서 로드하여 결과 확인
- [ ] Paired vs Unpaired 학습 결과 비교
- [ ] 모델 크기 < 10MB 확인

---

## Phase 3: 고급 스타일 전이 및 확산 모델 (6~8주)

### 목표
- D-LUT(확산 기반 LUT)로 더 정교한 색상 분포 학습
- StarGAN v2 통합으로 분위기 전체를 아우르는 스타일 전이
- 전체 시스템 통합 및 최적화

### 세부 작업

**3.1 D-LUT 구현 (3~4주)**
- Score-matching 기반 확산 모델 구현
- 색상 분포 학습용 forward/reverse diffusion process
- 학습된 변환 -> .cube 파일 추출 파이프라인
- Conditional generation: 참조 이미지 조건부 색상 분포 학습

**3.2 StarGAN v2 통합 (2~3주, 선택)**
- clovaai/stargan-v2 코드베이스 포크 및 커스터마이징
- 커스텀 도메인 정의 (색감/조명/분위기 카테고리)
- Style Encoder를 활용한 참조 이미지 기반 스타일 추출
- 경량화: Knowledge distillation으로 모델 크기 축소

**3.3 시스템 통합 (1~2주)**
- 통합 CLI/API 인터페이스
  - `analyze`: 레퍼런스 이미지 스타일 분석
  - `generate`: 필터 생성 (NILUT/3DLUT/D-LUT 선택)
  - `apply`: 대상 이미지에 필터 적용
  - `export`: .cube / .onnx / .pt 내보내기
- 모델 레지스트리: 학습된 모델/LUT 버전 관리
- 배치 처리 파이프라인

### 검증 방법
- [ ] D-LUT 생성 .cube vs Image-Adaptive 3D LUT .cube 품질 비교
- [ ] StarGAN v2 스타일 전이 FID/LPIPS 평가
- [ ] 전체 파이프라인 E2E 테스트 (입력 -> 스타일 분석 -> 필터 생성 -> 적용 -> 출력)
- [ ] 엣지 디바이스(모바일) 배포 테스트
- [ ] 사용자 A/B 테스트: AI 필터 vs 수동 보정 선호도

---

## 마일스톤 요약

| 단계 | 핵심 산출물 | 기간 | 품질 기준 충족 여부 |
|---|---|---|---|
| Phase 1 | NILUT 기반 다중 스타일 필터 시스템 | 4~6주 | ΔE < 1.0, 크기 < 1MB, 속도 < 16ms |
| Phase 2 | Image-Adaptive 3D LUT + .cube export | 4~6주 | ΔE < 1.0, 크기 < 10MB, 속도 < 2ms |
| Phase 3 | D-LUT + StarGAN v2 + 통합 시스템 | 6~8주 | 전체 품질 기준 + 상용 SW 호환 |

---

## 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|---|---|---|
| 학습 데이터 부족 | 모델 품질 저하 | Adobe FiveK 공개 데이터셋 활용, .cube 파일 기반 합성 데이터 생성 |
| CUDA 환경 비호환 | 개발 지연 | CPU fallback 구현, Apple MPS 백엔드 지원 |
| GAN 학습 불안정 | Phase 3 지연 | Phase 3의 StarGAN v2는 선택 사항으로 분류, LUT 계열에 집중 |
| 4K 실시간 목표 미달 | 사용성 저하 | CUDA 커널 최적화, 해상도 적응형 처리(저해상도 LUT 예측 -> 고해상도 적용) |
