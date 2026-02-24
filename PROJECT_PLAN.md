# D-Gate Project Plan

## 1. Project Overview

**D-Gate** is a production-grade diffusion model inference server designed to demonstrate mastery of modern generative-AI infrastructure. It serves as both a functional system and a portfolio piece aligned with senior AI/ML engineering roles (e.g., Leonardo.Ai AI Engineer).

### Target Role Alignment

| JD Requirement | D-Gate Coverage |
|---|---|
| Design & deploy transformer-based models | DiT/Flux backend alongside UNet |
| Quantisation & optimisation | INT8/FP8 calibration, TensorRT export |
| Model evaluation & benchmarking | CLIP score, FID, LPIPS, automated quality eval |
| Experiment tracking & reproducibility | Weights & Biases integration, run configs |
| Image-to-video generation | AnimateDiff, Stable Video Diffusion pipelines |
| Stylisation workflows | ControlNet, IP-Adapter, style presets |
| AWS cloud deployment | EKS manifests, GPU node pools, auto-scaling |
| PyTorch & diffusion models | Core engine built on diffusers + PyTorch |
| Docker & Kubernetes | Multi-stage Dockerfile, Kustomize overlays |
| MLOps & monitoring | Prometheus metrics, Grafana dashboards |

## 2. Architecture Overview

```
                         ┌─────────────────────┐
                         │   Load Balancer      │
                         │   (K8s Ingress)      │
                         └─────────┬────────────┘
                                   │
                         ┌─────────▼────────────┐
                         │   FastAPI Gateway     │
                         │   /v1/images/...      │
                         │   /v1/video/...       │
                         │   middleware (auth,    │
                         │   rate-limit, logging) │
                         └─────────┬────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │               │
           ┌────────▼───┐  ┌──────▼─────┐  ┌─────▼──────┐
           │  Request    │  │  Model     │  │ Experiment │
           │  Batcher    │  │  Registry  │  │ Tracker    │
           └────────┬───┘  └──────┬─────┘  └────────────┘
                    │              │
           ┌────────▼──────────────▼──────────────┐
           │          Engine Layer                  │
           │  ┌──────────┐ ┌───────────┐           │
           │  │  UNet    │ │  DiT/Flux │           │
           │  │  Backend │ │  Backend  │           │
           │  └────┬─────┘ └─────┬─────┘           │
           │       │             │                  │
           │  ┌────▼─────────────▼─────┐           │
           │  │  Quantization Engine   │           │
           │  │  (INT8 / FP8 / TRT)   │           │
           │  └────────────────────────┘           │
           │                                       │
           │  ┌─────────┐ ┌──────────┐ ┌────────┐ │
           │  │  LoRA   │ │ControlNet│ │  SVD/  │ │
           │  │ Manager │ │IP-Adapter│ │AnimDiff│ │
           │  └─────────┘ └──────────┘ └────────┘ │
           └───────────────────────────────────────┘
                            │
           ┌────────────────▼─────────────────┐
           │       Observability               │
           │  Prometheus → Grafana dashboards  │
           │  W&B experiment logging           │
           └──────────────────────────────────┘
```

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Framework | FastAPI, Uvicorn |
| ML Runtime | PyTorch 2.2+, diffusers, transformers |
| Optimization | TensorRT, torch.compile, bitsandbytes |
| Quantization | INT8 calibration, FP8 (Hopper), ONNX export |
| Video | AnimateDiff, Stable Video Diffusion |
| Style | ControlNet, IP-Adapter |
| Fine-tuning | PEFT (LoRA), DreamBooth |
| Evaluation | torchmetrics, clip-score, pytorch-fid, lpips |
| Experiment Tracking | Weights & Biases |
| Monitoring | Prometheus client, Grafana |
| Containerization | Docker (multi-stage, CUDA base) |
| Orchestration | Kubernetes, Kustomize |
| Cloud | AWS EKS, ECR, S3 |
| Testing | pytest, pytest-asyncio, httpx |

## 4. Phased Roadmap (20 Days)

### Phase 1: Core Serving (Days 1-3)

**Goal:** Functional image generation API with batching and metrics.

**Files:**
- `dgate/api/routes.py` — `/v1/images/generations` endpoint
- `dgate/api/models.py` — Pydantic request/response schemas
- `dgate/core/batcher.py` — Async request aggregation
- `dgate/core/config.py` — YAML config loader
- `dgate/core/metrics.py` — Prometheus counters/histograms
- `dgate/engine/diffusion.py` — UNet inference pipeline
- `dgate/engine/lora_manager.py` — Dynamic LoRA weight injection
- `main.py` — FastAPI app entrypoint
- `config.yaml` — Server + engine configuration
- `tests/test_api.py`, `tests/test_batcher.py`, `tests/test_engine.py`

**Exit Criteria:**
- [ ] `POST /v1/images/generations` returns a base64 image
- [ ] Dynamic batching aggregates 2+ concurrent requests
- [ ] Prometheus metrics exposed at `/metrics`
- [ ] LoRA weights inject in < 50ms
- [ ] All Phase 1 tests pass

**JD Alignment:** Inference pipelines, PyTorch, API design

---

### Phase 2: Optimization + Quality Evaluation (Days 4-7)

**Goal:** TensorRT acceleration, INT8/FP8 quantization, automated quality metrics.

**Files:**
- `dgate/engine/base.py` — Abstract `InferenceBackend` ABC
- `dgate/engine/tensorrt_backend.py` — TensorRT engine wrapper
- `dgate/engine/quantization.py` — INT8/FP8 calibration + export
- `dgate/evaluation/__init__.py`
- `dgate/evaluation/quality_metrics.py` — CLIP score, FID, LPIPS
- `dgate/evaluation/benchmark_runner.py` — Automated eval harness
- `dgate/evaluation/reference_dataset.py` — Reference image management
- `scripts/export_tensorrt.py` — UNet → TRT export script
- `scripts/calibrate_int8.py` — INT8 calibration data pipeline
- `scripts/run_quality_eval.py` — CLI for running evaluations
- `benchmark/quantization_bench.py` — Quantization impact benchmarks
- `benchmark/quality_eval_bench.py` — Quality metric benchmarks
- `tests/test_quantization.py`, `tests/test_evaluation.py`

**Exit Criteria:**
- [ ] TensorRT backend serves images with 2x+ speedup over PyTorch
- [ ] INT8 quantized model within 2% CLIP score of FP16 baseline
- [ ] FID, LPIPS, CLIP score computed on 1K reference set
- [ ] Benchmark results logged to `benchmark/results/`
- [ ] All Phase 2 tests pass

**JD Alignment:** Quantisation, model evaluation, benchmarking

---

### Phase 3: Multi-Architecture — DiT/Flux (Days 8-10)

**Goal:** Support transformer-based diffusion models alongside UNet.

**Files:**
- `dgate/engine/dit_backend.py` — DiT/Flux inference backend
- `dgate/core/model_registry.py` — Multi-model lifecycle manager
- `benchmark/dit_comparison_bench.py` — UNet vs DiT benchmarks
- `tests/test_dit.py`, `tests/test_model_registry.py`

**Exit Criteria:**
- [ ] DiT backend generates images via same API
- [ ] Model registry hot-swaps between UNet and DiT
- [ ] Comparative benchmarks (latency, quality, VRAM) logged
- [ ] All Phase 3 tests pass

**JD Alignment:** Transformer-based models, architecture design

---

### Phase 4: Style System + Fine-tuning (Days 11-13)

**Goal:** ControlNet conditioning, IP-Adapter style transfer, LoRA training.

**Files:**
- `dgate/style/__init__.py`
- `dgate/style/controlnet.py` — ControlNet integration
- `dgate/style/ip_adapter.py` — IP-Adapter style injection
- `dgate/style/style_presets.py` — Named style configurations
- `finetune/__init__.py`
- `finetune/train_lora.py` — LoRA training script
- `finetune/train_dreambooth.py` — DreamBooth training
- `finetune/dataset_prep.py` — Training data preprocessing
- `finetune/configs/` — Hyperparameter YAML files
- `tests/test_lora.py`, `tests/test_style.py`

**Exit Criteria:**
- [ ] ControlNet conditions on depth/canny/pose maps
- [ ] IP-Adapter applies reference image style
- [ ] Style presets selectable via API parameter
- [ ] LoRA training produces usable weights from 20 images
- [ ] All Phase 4 tests pass

**JD Alignment:** Stylisation workflows, fine-tuning

---

### Phase 5: Video Generation (Days 14-16)

**Goal:** Image-to-video and text-to-video pipelines.

**Files:**
- `dgate/video/__init__.py`
- `dgate/video/animatediff.py` — AnimateDiff pipeline
- `dgate/video/svd_pipeline.py` — Stable Video Diffusion
- `dgate/video/frame_interpolation.py` — Frame interpolation (RIFE)
- `dgate/api/routes.py` — Add `/v1/video/generations` endpoint
- `tests/test_video.py`

**Exit Criteria:**
- [ ] `POST /v1/video/generations` returns a video file
- [ ] AnimateDiff generates 16-frame clips
- [ ] SVD pipeline produces smooth image-to-video
- [ ] Frame interpolation doubles output FPS
- [ ] All Phase 5 tests pass

**JD Alignment:** Image-to-video generation

---

### Phase 6: Production Hardening (Days 17-18)

**Goal:** Experiment tracking, monitoring dashboards, auth middleware.

**Files:**
- `dgate/core/experiment_tracker.py` — W&B integration
- `dgate/api/middleware.py` — Auth, rate limiting, request logging
- `monitoring/prometheus/prometheus.yml` — Scrape configuration
- `monitoring/prometheus/alert_rules.yml` — Alert definitions
- `monitoring/grafana/dashboards/inference.json` — Main dashboard
- `monitoring/grafana/provisioning/datasources.yml` — Auto-provision
- `docker-compose.yml` — Full stack (app + prometheus + grafana)

**Exit Criteria:**
- [ ] W&B logs generation params, latency, quality scores per run
- [ ] Grafana dashboard shows throughput, latency p50/p95/p99, VRAM
- [ ] Prometheus alerts fire on latency > threshold
- [ ] API key auth rejects unauthorized requests
- [ ] Rate limiter enforces per-key quotas

**JD Alignment:** MLOps, experiment tracking, monitoring dashboards

---

### Phase 7: Deployment (Days 19-20)

**Goal:** Container images, Kubernetes manifests, AWS EKS deployment.

**Files:**
- `Dockerfile` — Multi-stage CUDA build
- `.dockerignore` — Exclude dev artifacts
- `k8s/base/deployment.yaml` — Base deployment
- `k8s/base/service.yaml` — ClusterIP service
- `k8s/base/kustomization.yaml` — Kustomize base
- `k8s/overlays/gpu/kustomization.yaml` — GPU node selector
- `k8s/overlays/cpu/kustomization.yaml` — CPU-only variant
- `k8s/overlays/aws-eks/kustomization.yaml` — EKS specifics
- `k8s/overlays/aws-eks/hpa.yaml` — Horizontal pod autoscaler
- `scripts/download_models.py` — Model download utility

**Exit Criteria:**
- [ ] Docker image builds and runs successfully
- [ ] `kustomize build k8s/overlays/gpu` produces valid manifests
- [ ] EKS overlay includes GPU tolerations, HPA, and S3 model mount
- [ ] Health check endpoints respond correctly
- [ ] All integration tests pass in container

**JD Alignment:** AWS, Kubernetes, Docker, cloud deployment

## 5. JD Gap Coverage Matrix

| JD Keyword | Phase | Implementation | Confidence |
|---|---|---|---|
| Transformer-based diffusion models | 3 | DiT/Flux backend | High |
| Quantisation (INT8, FP8) | 2 | Calibration pipeline + TRT | High |
| Model evaluation & benchmarking | 2 | CLIP, FID, LPIPS harness | High |
| Experiment tracking | 6 | W&B integration | High |
| Image-to-video | 5 | AnimateDiff + SVD | Medium |
| Stylisation workflows | 4 | ControlNet + IP-Adapter | High |
| Fine-tuning (LoRA, DreamBooth) | 4 | PEFT training scripts | High |
| AWS cloud deployment | 7 | EKS manifests + HPA | High |
| PyTorch & diffusers | 1 | Core engine | High |
| Docker & Kubernetes | 7 | Multi-stage + Kustomize | High |
| FastAPI serving | 1 | API gateway | High |
| Monitoring dashboards | 6 | Grafana + Prometheus | High |
| Batch processing | 1 | Async request batcher | High |
| VRAM optimization | 1 | VAE slicing/tiling | High |

## 6. Key Technical Decisions

### Why DiT/Flux alongside UNet?
UNet-based Stable Diffusion is the industry baseline, but the field is moving toward transformer-based architectures (Flux, Stable Diffusion 3). Supporting both demonstrates adaptability and understanding of architectural evolution.

### Why INT8 over pure TensorRT?
TensorRT provides the best single-engine performance, but INT8 quantization is more broadly applicable (works without NVIDIA-specific tooling) and directly maps to the "quantisation" JD requirement. We support both paths.

### Why W&B over MLflow?
Leonardo.Ai's stack likely uses W&B or similar. W&B provides better experiment comparison UX, team collaboration features, and GPU utilization tracking out of the box. MLflow remains an alternative via config toggle.

### Why Kustomize over Helm?
Kustomize is built into kubectl, requires no template engine, and produces readable YAML patches. For a project of this scope, Kustomize overlays are simpler to maintain and review.

### Why AnimateDiff + SVD?
AnimateDiff extends existing SD/SDXL pipelines (reuses our UNet backend), while SVD is the state-of-the-art for image-to-video. Together they cover text-to-video and image-to-video use cases.

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| VRAM constraints (12GB) | Cannot load large models | VAE tiling, model offloading, quantization |
| TensorRT build failures | Blocked optimization path | Fallback to torch.compile, separate TRT CI stage |
| Video generation OOM | Pipeline crashes | Frame-by-frame generation, aggressive offloading |
| DiT model availability | Cannot demonstrate transformer backend | Pin known-good model revisions, mock backends for tests |
| W&B API instability | Experiment data lost | Local JSON fallback, async logging |
| Large Docker images | Slow deploys | Multi-stage builds, separate base/app layers |

## 8. Directory Structure

```
D-Gate/
├── main.py
├── config.yaml
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── README.md
├── PROJECT_PLAN.md
├── dgate/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── middleware.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── batcher.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   ├── model_registry.py
│   │   └── experiment_tracker.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── diffusion.py
│   │   ├── dit_backend.py
│   │   ├── lora_manager.py
│   │   ├── tensorrt_backend.py
│   │   └── quantization.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── quality_metrics.py
│   │   ├── benchmark_runner.py
│   │   └── reference_dataset.py
│   ├── style/
│   │   ├── __init__.py
│   │   ├── controlnet.py
│   │   ├── ip_adapter.py
│   │   └── style_presets.py
│   └── video/
│       ├── __init__.py
│       ├── animatediff.py
│       ├── svd_pipeline.py
│       └── frame_interpolation.py
├── finetune/
│   ├── __init__.py
│   ├── train_lora.py
│   ├── train_dreambooth.py
│   ├── dataset_prep.py
│   └── configs/
│       └── lora_default.yaml
├── scripts/
│   ├── export_tensorrt.py
│   ├── calibrate_int8.py
│   ├── download_models.py
│   └── run_quality_eval.py
├── benchmark/
│   ├── memory_bench.py
│   ├── scheduler_bench.py
│   ├── quantization_bench.py
│   ├── quality_eval_bench.py
│   ├── dit_comparison_bench.py
│   └── results/
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_batcher.py
│   ├── test_engine.py
│   ├── test_lora.py
│   ├── test_quantization.py
│   ├── test_evaluation.py
│   ├── test_dit.py
│   ├── test_style.py
│   ├── test_video.py
│   └── test_model_registry.py
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── inference.json
│   │   └── provisioning/
│   │       └── datasources.yml
│   └── prometheus/
│       ├── prometheus.yml
│       └── alert_rules.yml
└── k8s/
    ├── base/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── kustomization.yaml
    └── overlays/
        ├── gpu/
        │   └── kustomization.yaml
        ├── cpu/
        │   └── kustomization.yaml
        └── aws-eks/
            ├── kustomization.yaml
            └── hpa.yaml
```
