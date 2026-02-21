"""
QC Gates: 품질 검증 게이트
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Optional dependencies
try:
    import pyloudnorm as pyln
    LOUDNORM_AVAILABLE = True
except ImportError:
    LOUDNORM_AVAILABLE = False


class TechnicalGate:
    """QC-1: 기술 검증 (클리핑, LUFS, True Peak, DC offset)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: QC Gate 설정
                {
                    "clipping_ratio_max": 0.001,
                    "lufs_range": {"min": -23, "max": -13},
                    "true_peak_max_db": -1.0,
                    "dc_offset_max": 0.01
                }
        """
        config = config or {}
        self.clipping_ratio_max = config.get("clipping_ratio_max", 0.001)
        self.lufs_min = config.get("lufs_range", {}).get("min", -23)
        self.lufs_max = config.get("lufs_range", {}).get("max", -13)
        self.true_peak_max_db = config.get("true_peak_max_db", -1.0)
        self.dc_offset_max = config.get("dc_offset_max", 0.01)

    def check(self, audio: np.ndarray, sr: int) -> Tuple[bool, Dict[str, Any]]:
        """
        기술 검증

        Args:
            audio: 오디오 (shape: [samples] or [samples, channels])
            sr: 샘플레이트

        Returns:
            (passed: bool, results: Dict)
        """
        results = {}

        # Clipping 검사
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / audio.size
        results["clipping_ratio"] = float(clipping_ratio)

        # LUFS / True Peak
        if LOUDNORM_AVAILABLE:
            meter = pyln.Meter(sr)
            # Reshape to [samples, channels]
            if audio.ndim == 1:
                audio_2d = audio.reshape(-1, 1)
            else:
                audio_2d = audio

            lufs = meter.integrated_loudness(audio_2d)
            results["lufs"] = float(lufs)
        else:
            # Fallback: RMS-based approximation
            rms = np.sqrt(np.mean(audio ** 2))
            lufs_approx = 20 * np.log10(rms + 1e-10) - 0.691  # -0.691 = 0 dBFS to LUFS offset
            results["lufs"] = float(lufs_approx)
            lufs = lufs_approx

        # True Peak
        true_peak = np.max(np.abs(audio))
        true_peak_db = 20 * np.log10(true_peak + 1e-10)
        results["true_peak_db"] = float(true_peak_db)

        # DC offset
        dc_offset = np.mean(audio)
        results["dc_offset"] = float(dc_offset)

        # Silence holes (optional)
        silence_threshold = 0.001
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / audio.size
        results["silence_ratio"] = float(silence_ratio)

        # 판정
        passed = (
            clipping_ratio < self.clipping_ratio_max and
            self.lufs_min <= lufs <= self.lufs_max and
            true_peak_db < self.true_peak_max_db and
            abs(dc_offset) < self.dc_offset_max
        )

        results["passed"] = passed
        results["failures"] = []
        if clipping_ratio >= self.clipping_ratio_max:
            results["failures"].append(f"Clipping ratio {clipping_ratio:.4f} exceeds {self.clipping_ratio_max}")
        if not (self.lufs_min <= lufs <= self.lufs_max):
            results["failures"].append(f"LUFS {lufs:.2f} outside range [{self.lufs_min}, {self.lufs_max}]")
        if true_peak_db >= self.true_peak_max_db:
            results["failures"].append(f"True peak {true_peak_db:.2f} dB exceeds {self.true_peak_max_db} dB")
        if abs(dc_offset) >= self.dc_offset_max:
            results["failures"].append(f"DC offset {dc_offset:.4f} exceeds {self.dc_offset_max}")

        return passed, results


class LocalityGate:
    """QC-2: NDE Locality - 마스크 밖 변화량 측정"""

    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: 허용 변화량 (예: 5% = 0.05)
        """
        self.threshold = threshold

    def check(
        self,
        original: np.ndarray,
        edited: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        OutMaskDelta: 마스크 밖 영역의 변화량 측정

        Args:
            original: 원본 오디오
            edited: 편집된 오디오
            mask: 편집 마스크 (1 = 편집 영역, 0 = 보존 영역)
                  shape은 original과 동일하거나 broadcastable

        Returns:
            (passed: bool, results: Dict)
        """
        # Shape 검증
        if original.shape != edited.shape:
            raise ValueError(f"Shape mismatch: {original.shape} != {edited.shape}")

        # Mask broadcast (필요 시)
        if mask.shape != original.shape:
            mask = np.broadcast_to(mask, original.shape)

        # 마스크 반전 (밖 영역)
        out_mask = 1 - mask

        # 마스크 밖 차이 계산
        diff = np.abs(edited - original)
        out_mask_diff = diff * out_mask

        # 평균 변화량 (원본 RMS 대비)
        original_rms = np.sqrt(np.mean(original ** 2))
        if original_rms < 1e-10:
            # 원본이 거의 무음 → 절대 차이 기준
            out_mask_delta = np.mean(out_mask_diff)
        else:
            out_mask_delta = np.mean(out_mask_diff) / original_rms

        # 최대 변화량
        out_mask_max_delta = np.max(out_mask_diff) / (original_rms + 1e-10)

        results = {
            "out_mask_delta": float(out_mask_delta),
            "out_mask_max_delta": float(out_mask_max_delta),
            "threshold": self.threshold
        }

        passed = out_mask_delta < self.threshold

        results["passed"] = passed
        if not passed:
            results["failures"] = [f"OutMaskDelta {out_mask_delta:.4f} exceeds threshold {self.threshold}"]
        else:
            results["failures"] = []

        return passed, results


class SimilarityGate:
    """QC-3: Similarity Risk - 라이브러리 near-duplicate 탐지"""

    def __init__(self, threshold: float = 0.95, reference_library_path: Optional[str] = None):
        """
        Args:
            threshold: 유사도 임계값 (95% 이상 시 플래그)
            reference_library_path: 참조 라이브러리 경로
        """
        self.threshold = threshold
        self.reference_library_path = reference_library_path
        self.reference_embeddings = self._load_reference_library()

    def _load_reference_library(self) -> Optional[np.ndarray]:
        """참조 라이브러리 임베딩 로드 (Placeholder)"""
        # 실제 구현: CLAP/MusicGen embedding 로드
        return None

    def check(self, audio: np.ndarray, sr: int) -> Tuple[bool, Dict[str, Any]]:
        """
        유사도 검증

        Args:
            audio: 검사 대상 오디오
            sr: 샘플레이트

        Returns:
            (passed: bool, results: Dict)
        """
        # Placeholder: 실제로는 CLAP/MusicGen으로 embedding 추출
        # 그 다음 cosine similarity 계산

        if self.reference_embeddings is None:
            # 참조 라이브러리 없음 → 통과
            return True, {
                "passed": True,
                "max_similarity": 0.0,
                "flagged_references": [],
                "note": "Reference library not loaded"
            }

        # TODO: Embedding 추출 및 유사도 계산
        max_similarity = 0.0
        flagged_references = []

        passed = max_similarity < self.threshold

        results = {
            "passed": passed,
            "max_similarity": float(max_similarity),
            "flagged_references": flagged_references,
            "threshold": self.threshold
        }

        if not passed:
            results["failures"] = [f"Similarity {max_similarity:.4f} exceeds threshold {self.threshold}"]
        else:
            results["failures"] = []

        return passed, results


class QCPipeline:
    """전체 QC Gate 파이프라인"""

    def __init__(self, policy: Dict[str, Any]):
        """
        Args:
            policy: policy.v1.schema.json 형식의 정책
        """
        qc_gates_config = policy.get("qc_gates", {})

        # Technical Gate
        tech_config = qc_gates_config.get("technical", {})
        self.technical_gate = TechnicalGate(tech_config) if tech_config.get("enabled", True) else None

        # Locality Gate
        locality_config = qc_gates_config.get("locality", {})
        self.locality_gate = LocalityGate(
            threshold=locality_config.get("out_mask_delta_max", 0.05)
        ) if locality_config.get("enabled", True) else None

        # Similarity Gate
        similarity_config = qc_gates_config.get("similarity", {})
        self.similarity_gate = SimilarityGate(
            threshold=similarity_config.get("similarity_threshold", 0.95),
            reference_library_path=similarity_config.get("reference_library_path")
        ) if similarity_config.get("enabled", False) else None

    def run(
        self,
        audio: np.ndarray,
        sr: int,
        original: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        전체 QC Gate 실행

        Args:
            audio: 검사 대상 오디오
            sr: 샘플레이트
            original: 원본 오디오 (Locality Gate용, optional)
            mask: 편집 마스크 (Locality Gate용, optional)

        Returns:
            (passed: bool, all_results: Dict)
        """
        all_results = {}
        all_passed = True

        # QC-1: Technical
        if self.technical_gate:
            passed, results = self.technical_gate.check(audio, sr)
            all_results["technical"] = results
            all_passed = all_passed and passed

        # QC-2: Locality
        if self.locality_gate and original is not None and mask is not None:
            passed, results = self.locality_gate.check(original, audio, mask)
            all_results["locality"] = results
            all_passed = all_passed and passed

        # QC-3: Similarity
        if self.similarity_gate:
            passed, results = self.similarity_gate.check(audio, sr)
            all_results["similarity"] = results
            all_passed = all_passed and passed

        return all_passed, all_results
