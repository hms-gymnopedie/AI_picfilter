# AI 모델 상세 비교표

## 1. 종합 비교

| 평가 항목 | Neural Style Transfer | Image-Adaptive 3D LUT | NILUT | StarGAN v2 | D-LUT |
|---|---|---|---|---|---|
| **색상 정확도 (ΔE)** | ~2.0-5.0 (주관적) | < 1.0 | < 1.0 | ~3.0-8.0 | < 1.5 |
| **추론 속도 (4K, GPU)** | 5-30초 | < 2ms | 5-15ms | 100-500ms | 50-200ms (학습), <2ms (LUT 적용) |
| **추론 속도 (4K, CPU)** | 수분 | 10-50ms | 30-100ms | 수초 | 수초 (학습), 10-50ms (LUT 적용) |
| **파라미터 수** | ~14.7M (VGG-19, frozen) | ~0.6M (CNN) + LUT | ~0.01-0.05M | ~70-100M | ~5-20M (학습) |
| **모델 파일 크기** | ~550MB (VGG-19) | 0.5-5MB | < 0.25MB | 50-300MB | 10-50MB (학습), <1MB (.cube) |
| **구현 난이도** | 3/5 | 2/5 | 1/5 | 4/5 | 5/5 |
| **오픈소스 활용** | 제한적 (Torch/Lua 레거시) | 높음 (PyTorch, 잘 정리됨) | 높음 (PyTorch, 튜토리얼 포함) | 높음 (PyTorch, 사전학습 제공) | 제한적 (연구 초기 단계) |
| **Phase 적합성** | 참고용 | Phase 2 | Phase 1 (MVP) | Phase 3 (선택) | Phase 3 |
| **권장 GPU** | NVIDIA GTX 1080+ | 불필요 (CPU 가능) | 불필요 (CPU 가능) | NVIDIA RTX 3080+ | NVIDIA RTX 3080+ |
| **권장 RAM** | 16GB+ | 4GB+ | 2GB+ | 16GB+ | 16GB+ |
| **VRAM** | 8GB+ | 1GB (학습 시) | 1GB (학습 시) | 12GB+ | 8GB+ |

---

## 2. 항목별 상세 분석

### 2.1 색상 정확도 (CIE76 ΔE)

| 모델 | 예상 ΔE | 비고 |
|---|---|---|
| **NILUT** | < 1.0 | 연속 함수로 보간 오차 없음. 512개 스타일에서도 ΔE < 1.0 달성 보고 |
| **Image-Adaptive 3D LUT** | < 1.0 | Basis LUT 블렌딩으로 고정밀. 그리드 크기(33 or 64)에 따라 정밀도 변동 |
| **D-LUT** | < 1.5 | 확산 과정의 확률적 특성으로 미세 변동 가능. 다수 샘플링으로 보정 |
| **Neural Style Transfer** | 2.0-5.0 | 최적화 기반이므로 수렴 정도에 따라 달라짐. 정량적 ΔE 최적화가 목적이 아님 |
| **StarGAN v2** | 3.0-8.0 | GAN 특성상 정확한 색상 재현보다 시각적 자연스러움에 초점 |

### 2.2 추론 속도

| 모델 | GPU (4K) | CPU (4K) | 속도 결정 요인 |
|---|---|---|---|
| **Image-Adaptive 3D LUT** | < 2ms | 10-50ms | LUT lookup은 O(1). CNN은 저해상도 입력만 처리 |
| **NILUT** | 5-15ms | 30-100ms | 픽셀별 MLP 추론. 배치 처리로 병렬화 가능 |
| **D-LUT** | < 2ms (적용) | 10-50ms (적용) | 학습 후 .cube export하면 일반 LUT와 동일 속도 |
| **StarGAN v2** | 100-500ms | 2-10초 | 전체 이미지를 인코더-디코더로 처리 |
| **Neural Style Transfer** | 5-30초 | 수분 | 반복 최적화(500-1000 iterations) 필요 |

### 2.3 모델 크기

| 모델 | 파라미터 수 | 디스크 크기 | 10MB 목표 충족 |
|---|---|---|---|
| **NILUT** | ~10K-50K | < 0.25MB | O (초과 달성) |
| **Image-Adaptive 3D LUT** | ~600K (CNN + LUT) | 0.5-5MB | O |
| **D-LUT** | ~5-20M (학습 모델) | .cube export < 1MB | O (.cube 기준) |
| **Neural Style Transfer** | ~14.7M (VGG-19) | ~550MB | X |
| **StarGAN v2** | ~70-100M | 50-300MB | X |

### 2.4 구현 난이도

| 모델 | 난이도 | 근거 |
|---|---|---|
| **NILUT** | 1/5 | 단순 MLP. PyTorch 기초만으로 구현 가능. mv-lab/nilut에 fit.py, 튜토리얼 노트북 제공 |
| **Image-Adaptive 3D LUT** | 2/5 | CNN + trilinear interpolation 구현 필요. HuiZeng 레포에 학습 스크립트 완비 |
| **Neural Style Transfer** | 3/5 | VGG feature 추출, Gram matrix, 라플라시안 행렬 구현. 레거시 Torch/Lua 코드를 PyTorch로 포팅 필요 |
| **StarGAN v2** | 4/5 | 복잡한 GAN 학습(Generator, Discriminator, Style Encoder, Mapping Network). 학습 불안정성 대응 필요 |
| **D-LUT** | 5/5 | Score-matching, diffusion process 이해 필요. 공개 구현체가 제한적이며 논문 기반 자체 구현 부담 |

### 2.5 오픈소스 활용 가능성

| 모델 | 레포지토리 | 라이선스 | 활용도 | 비고 |
|---|---|---|---|---|
| **NILUT** | mv-lab/nilut | MIT | 높음 | fit.py, 멀티블렌드 노트북, 사전학습 모델 제공 |
| **Image-Adaptive 3D LUT** | HuiZeng/Image-Adaptive-3DLUT | Apache 2.0 | 높음 | Paired/Unpaired 학습 코드, 사전학습 모델 제공 |
| **StarGAN v2** | clovaai/stargan-v2 | CC BY-NC 4.0 | 중간 | 사전학습 모델 제공. 상업적 사용 시 라이선스 확인 필요 |
| **Neural Style Transfer** | luanfujun/deep-photo-styletransfer | MIT | 낮음 | Torch/Lua + MATLAB 기반. PyTorch 포팅 필요 |
| **D-LUT** | (연구 초기) | - | 낮음 | 공개 구현체가 제한적. 논문 참고하여 자체 구현 필요 |

---

## 3. Phase별 적합성 매트릭스

| 모델 | Phase 1 (MVP) | Phase 2 (품질 개선) | Phase 3 (고급 기능) |
|---|---|---|---|
| **NILUT** | **핵심** — MVP의 메인 모델 | 보조 — 경량 배포용 유지 | 보조 — 멀티스타일 블렌딩 UI |
| **Image-Adaptive 3D LUT** | 참고 | **핵심** — 적응형 필터 + .cube export | 유지 — 실시간 처리 백엔드 |
| **D-LUT** | - | - | **핵심** — 고품질 .cube 생성 |
| **StarGAN v2** | - | - | **선택** — 분위기 전체 전이 |
| **Neural Style Transfer** | - | 참고/실험 | 참고 — 비교 기준선 |

---

## 4. 권장 하드웨어 요구사항

### Phase 1 (MVP) — NILUT 중심

| 항목 | 최소 | 권장 |
|---|---|---|
| CPU | Apple M1 / Intel i5 | Apple M2 Pro / Intel i7 |
| RAM | 8GB | 16GB |
| GPU | 불필요 (CPU 학습 가능) | NVIDIA GTX 1060 / Apple MPS |
| 저장 공간 | 10GB | 50GB (데이터셋 포함) |
| OS | macOS 13+ / Ubuntu 20.04+ | macOS 14+ / Ubuntu 22.04+ |

### Phase 2 — Image-Adaptive 3D LUT 추가

| 항목 | 최소 | 권장 |
|---|---|---|
| CPU | Apple M1 Pro / Intel i7 | Apple M2 Pro / AMD Ryzen 7 |
| RAM | 16GB | 32GB |
| GPU | NVIDIA GTX 1080 / Apple MPS | NVIDIA RTX 3060 |
| VRAM | 4GB | 8GB |
| 저장 공간 | 50GB | 200GB |

### Phase 3 — D-LUT, StarGAN v2 추가

| 항목 | 최소 | 권장 |
|---|---|---|
| CPU | Intel i7 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| RAM | 32GB | 64GB |
| GPU | NVIDIA RTX 3080 | NVIDIA RTX 4090 / A100 |
| VRAM | 12GB | 24GB+ |
| 저장 공간 | 200GB | 500GB+ (대규모 데이터셋) |

---

## 5. 결론 및 권장 사항

1. **Phase 1은 NILUT로 시작**: 구현 난이도 최저, 품질 기준 즉시 충족, 하드웨어 요구 최소
2. **Phase 2에서 Image-Adaptive 3D LUT 추가**: 실시간 4K 처리(<2ms)와 .cube export 확보
3. **Phase 3의 StarGAN v2는 선택 사항**: 라이선스(CC BY-NC 4.0)와 높은 리소스 요구를 고려하여 필요 시에만 도입
4. **D-LUT는 연구 트래킹 유지**: 공개 구현체가 성숙하면 도입. 그 전까지는 Image-Adaptive 3D LUT의 .cube export로 대체
