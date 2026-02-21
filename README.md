# apex-audio

**APEX Audio Nondestructive Edition Platform**

오디오 비파괴 편집을 위한 Two-Plane 아키텍처 플랫폼

## 핵심 특징

### NDE-First 설계
- **Edit Graph**: Git patch 스타일의 append-only 편집 히스토리
- **CRDT**: 협업 편집 지원 (conflict resolution)
- **Reproducibility**: seed + engine_version으로 결정적 재현

### Two-Plane 아키텍처
- **Control Plane**: 오케스트레이션, 권한, 정책 (TypeScript)
- **Render Plane**: GPU 렌더링, DSP, AI (Python + CUDA)

### QC Gate (4단계)
1. **Policy Gate**: 고객사 정책 검증 (enhancement_only 등)
2. **Technical Gate**: 클리핑, LUFS, True Peak, DC offset
3. **Locality Gate**: 마스크 밖 변화량 측정 (OutMaskDelta)
4. **Similarity Gate**: 라이브러리 near-duplicate 탐지

### 계약 기반 통합
- apex-core contracts (SSoT)
- SSE 프로그레스 스트리밍
- NATS IPC (Reliable Job Plane)

## 아키텍처

```
apex-core (contracts SSoT)
  ├── audio.v1.schema.json
  ├── editgraph.v1.schema.json
  ├── artifact.v1.schema.json
  └── policy.v1.schema.json

apex-audio/
  ├── packages/
  │   ├── control-plane/        # TS: HTTP/SSE, auth
  │   ├── nde-core/             # Python: EditGraph, CRDT
  │   ├── render-workers/       # GPU: ACE-Step + DSP
  │   ├── qc-metrics/           # QC Gates
  │   └── artifact-packager/    # 결과물 패키징
  ├── plugins/
  │   ├── dsp/                  # EQ, Comp, Reverb
  │   ├── ai/                   # Repaint, Cover
  │   └── analyzers/            # LUFS, Spectrum
  └── tests/
      ├── unit/
      ├── integration/
      ├── e2e/
      └── determinism/          # 재현성 검증
```

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt
cd packages/control-plane && npm install

# Docker Compose
docker-compose up -d

# 테스트
python -m pytest tests/
npm test
```

## API Endpoints

### NDE 세션
- `POST /api/nde/session/create` - 세션 생성
- `POST /api/nde/edit/apply` - 편집 노드 추가 (SSE)
- `POST /api/nde/render/preview` - 프리뷰 렌더 (SSE)
- `POST /api/nde/render/final` - 최종 렌더 (SSE + QC)
- `GET /api/nde/history/:session_id` - Edit Graph 조회
- `GET /api/nde/artifact/:session_id` - Artifact 다운로드

## 개발 가이드

### Determinism 보장
```python
# Seed 고정
node = EditNode(
    node_type="repaint",
    params={"intensity": 0.5},
    seed=42,
    engine_version="1.0.0"
)

# 재현 가능
result1 = renderer.render_node(audio, node)
result2 = renderer.render_node(audio, node)
assert np.allclose(result1, result2)
```

### CRDT Edit Graph
```python
# 협업 편집
graph_a = EditGraph(source_hash="abc...")
graph_a.append_node(node1)

graph_b = EditGraph(source_hash="abc...")
graph_b.append_node(node2)

# 병합
merged = graph_a.merge(graph_b)
```

### Policy Gate
```python
policy = {
    "allowed_operations": {
        "enhancement_only": True,
        "inpaint_allowed": False
    }
}

# 정책 위반 검사
passed, reason = policy_gate.check(node, policy)
if not passed:
    raise PolicyViolationError(reason)
```

## 라이선스

MIT
