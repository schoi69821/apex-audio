"""
Deterministic Renderer: 재현 가능한 렌더링 엔진
"""
import numpy as np
import hashlib
import os
import pickle
from typing import Optional, Dict, Any
from pathlib import Path

# Optional GPU support
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


class DeterministicRenderer:
    """재현 가능한 렌더링 엔진"""

    def __init__(self, engine_version: str, cache_dir: str):
        """
        Args:
            engine_version: 엔진 버전 (semantic versioning)
            cache_dir: Segment cache 디렉토리
        """
        self.engine_version = engine_version
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache 통계
        self.cache_hits = 0
        self.cache_misses = 0

    def render_node(self, audio: np.ndarray, node: Any, sr: int = 48000) -> np.ndarray:
        """
        단일 노드 적용 (deterministic)

        Args:
            audio: 입력 오디오 (shape: [samples] or [samples, channels])
            node: EditNode 객체
            sr: 샘플레이트

        Returns:
            편집된 오디오 (동일 shape)
        """
        # Seed 고정 (재현성)
        if node.seed is not None:
            np.random.seed(node.seed)
            if GPU_AVAILABLE:
                torch.manual_seed(node.seed)
                torch.cuda.manual_seed_all(node.seed)
                # Deterministic operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # 캐시 확인
        cache_key = self._cache_key(node)
        if cached := self._load_cache(cache_key, audio.shape):
            self.cache_hits += 1
            return cached

        self.cache_misses += 1

        # 렌더 실행
        result = self._apply_node(audio, node, sr)

        # 캐시 저장
        self._save_cache(cache_key, result)

        return result

    def _cache_key(self, node: Any) -> str:
        """
        Segment cache key

        재현성 보장을 위해:
        - node.hash() (params + seed + engine_version)
        - self.engine_version
        기반으로 키 생성
        """
        node_hash = node.hash() if hasattr(node, 'hash') else str(node.node_id)
        combined = f"{node_hash}:{self.engine_version}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _load_cache(self, cache_key: str, expected_shape: tuple) -> Optional[np.ndarray]:
        """캐시 로드"""
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if not cache_path.exists():
            return None

        try:
            cached = np.load(cache_path)
            # Shape 검증
            if cached.shape != expected_shape:
                return None
            return cached
        except Exception:
            return None

    def _save_cache(self, cache_key: str, result: np.ndarray):
        """캐시 저장"""
        cache_path = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_path, result)
        except Exception as e:
            print(f"Cache save failed: {e}")

    def _apply_node(self, audio: np.ndarray, node: Any, sr: int) -> np.ndarray:
        """
        실제 DSP/AI 적용

        Args:
            audio: 입력 오디오
            node: EditNode
            sr: 샘플레이트

        Returns:
            편집된 오디오
        """
        node_type = node.node_type
        params = node.params

        if node_type == "dsp":
            return self._apply_dsp(audio, params, sr)
        elif node_type == "repaint":
            return self._apply_repaint(audio, params, node.mask, sr)
        elif node_type == "cover":
            return self._apply_cover(audio, params, sr)
        elif node_type == "split":
            return self._apply_split(audio, params, sr)
        elif node_type == "align":
            return self._apply_align(audio, params, sr)
        elif node_type == "mix":
            return self._apply_mix(audio, params, sr)
        elif node_type == "export":
            return audio  # No-op
        else:
            raise ValueError(f"Unknown node_type: {node_type}")

    def _apply_dsp(self, audio: np.ndarray, params: Dict[str, Any], sr: int) -> np.ndarray:
        """DSP 처리 (EQ, Compressor, Reverb 등)"""
        result = audio.copy()

        # EQ
        if "eq" in params:
            result = self._apply_eq(result, params["eq"], sr)

        # Compressor
        if "compressor" in params:
            result = self._apply_compressor(result, params["compressor"])

        # Time stretch
        if "time_stretch" in params:
            result = self._apply_time_stretch(result, params["time_stretch"], sr)

        # Pitch shift
        if "pitch_shift" in params:
            result = self._apply_pitch_shift(result, params["pitch_shift"], sr)

        return result

    def _apply_eq(self, audio: np.ndarray, eq_params: Dict, sr: int) -> np.ndarray:
        """간단한 EQ (실제로는 scipy.signal 사용)"""
        # Placeholder: 실제 구현에서는 biquad filter 체인
        return audio

    def _apply_compressor(self, audio: np.ndarray, comp_params: Dict) -> np.ndarray:
        """간단한 컴프레서"""
        threshold = comp_params.get("threshold", -20)
        ratio = comp_params.get("ratio", 4.0)

        # Simple peak compressor
        threshold_linear = 10 ** (threshold / 20)
        above_threshold = np.abs(audio) > threshold_linear

        result = audio.copy()
        result[above_threshold] = (
            np.sign(audio[above_threshold]) * threshold_linear +
            (np.abs(audio[above_threshold]) - threshold_linear) / ratio
        )

        return result

    def _apply_time_stretch(self, audio: np.ndarray, rate: float, sr: int) -> np.ndarray:
        """시간 늘이기/줄이기 (Placeholder)"""
        # 실제로는 librosa.effects.time_stretch 또는 rubberband 사용
        # 여기서는 리샘플링으로 근사
        new_length = int(len(audio) / rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    def _apply_pitch_shift(self, audio: np.ndarray, semitones: int, sr: int) -> np.ndarray:
        """피치 시프트 (Placeholder)"""
        # 실제로는 librosa.effects.pitch_shift 사용
        return audio

    def _apply_repaint(self, audio: np.ndarray, params: Dict, mask: Optional[Dict], sr: int) -> np.ndarray:
        """
        AI Inpaint (노이즈 제거, 결함 수정)

        실제 구현에서는:
        - ACE-Step-1.5 모델 사용
        - mask 영역만 inpainting
        - 나머지는 원본 유지 (locality 보장)
        """
        # Placeholder
        return audio

    def _apply_cover(self, audio: np.ndarray, params: Dict, sr: int) -> np.ndarray:
        """AI Cover 생성 (Placeholder)"""
        return audio

    def _apply_split(self, audio: np.ndarray, params: Dict, sr: int) -> np.ndarray:
        """소스 분리 (Placeholder)"""
        return audio

    def _apply_align(self, audio: np.ndarray, params: Dict, sr: int) -> np.ndarray:
        """오디오 정렬 (Placeholder)"""
        return audio

    def _apply_mix(self, audio: np.ndarray, params: Dict, sr: int) -> np.ndarray:
        """믹싱 (Placeholder)"""
        return audio

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": hit_ratio,
            "cache_dir_size_mb": sum(f.stat().st_size for f in self.cache_dir.glob("*.npy")) / 1024 / 1024
        }

    def clear_cache(self):
        """캐시 클리어"""
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        self.cache_hits = 0
        self.cache_misses = 0
