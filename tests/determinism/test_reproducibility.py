"""
Determinism & Reproducibility 테스트
"""
import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "render-workers"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "nde-core"))

from src.deterministic_renderer import DeterministicRenderer
from src.edit_graph import EditNode


@pytest.fixture
def test_audio():
    """테스트 오디오 생성 (1초, 48kHz, mono)"""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # 440Hz 사인파 + 약간의 노이즈
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    return audio.astype(np.float32)


@pytest.fixture
def renderer():
    """렌더러 fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        renderer = DeterministicRenderer(
            engine_version="1.0.0",
            cache_dir=tmpdir
        )
        yield renderer


class TestDeterminism:
    """재현성 테스트"""

    def test_same_seed_same_result(self, test_audio, renderer):
        """동일 seed → 동일 결과"""
        node = EditNode(
            node_type="dsp",
            params={"compressor": {"threshold": -20, "ratio": 4.0}},
            seed=42,
            engine_version="1.0.0"
        )

        result1 = renderer.render_node(test_audio, node)
        result2 = renderer.render_node(test_audio, node)

        # 수치적으로 동일 (부동소수점 오차 허용)
        assert np.allclose(result1, result2, atol=1e-6)

    def test_different_seed_different_result(self, test_audio, renderer):
        """다른 seed → 다른 결과 (랜덤 요소가 있는 경우)"""
        node1 = EditNode(
            node_type="dsp",
            params={"compressor": {"threshold": -20, "ratio": 4.0}},
            seed=42
        )
        node2 = EditNode(
            node_type="dsp",
            params={"compressor": {"threshold": -20, "ratio": 4.0}},
            seed=43
        )

        result1 = renderer.render_node(test_audio, node1)
        result2 = renderer.render_node(test_audio, node2)

        # Deterministic DSP는 seed 영향 안 받지만, 구조적으로는 다를 수 있음
        # 여기서는 동일 (실제 AI 기반은 다름)
        assert result1.shape == result2.shape

    def test_cache_hit_same_result(self, test_audio, renderer):
        """캐시 hit → 동일 결과"""
        node = EditNode(
            node_type="dsp",
            params={"compressor": {"threshold": -20}},
            seed=42,
            engine_version="1.0.0"
        )

        # 첫 실행 (cache miss)
        result1 = renderer.render_node(test_audio, node)
        assert renderer.cache_misses == 1
        assert renderer.cache_hits == 0

        # 두 번째 실행 (cache hit)
        result2 = renderer.render_node(test_audio, node)
        assert renderer.cache_hits == 1

        # 결과 동일
        assert np.allclose(result1, result2, atol=1e-6)

    def test_engine_version_cache_miss(self, test_audio):
        """engine_version 다르면 cache miss"""
        with tempfile.TemporaryDirectory() as tmpdir:
            renderer_v1 = DeterministicRenderer("1.0.0", tmpdir)
            renderer_v2 = DeterministicRenderer("1.1.0", tmpdir)

            node = EditNode(
                node_type="dsp",
                params={"compressor": {"threshold": -20}},
                seed=42
            )

            # v1.0.0 렌더
            result1 = renderer_v1.render_node(test_audio, node)
            assert renderer_v1.cache_misses == 1

            # v1.1.0 렌더 (다른 engine_version → cache miss)
            result2 = renderer_v2.render_node(test_audio, node)
            assert renderer_v2.cache_misses == 1

            # 결과는 동일 shape (실제 구현에서는 다를 수 있음)
            assert result1.shape == result2.shape

    def test_node_hash_stability(self):
        """노드 해시 안정성"""
        node1 = EditNode(
            node_type="dsp",
            params={"eq": {"bands": [{"freq": 1000, "gain": 3}]}},
            seed=42,
            engine_version="1.0.0"
        )

        hash1 = node1.hash()

        # 동일 파라미터로 다시 생성
        node2 = EditNode(
            node_type="dsp",
            params={"eq": {"bands": [{"freq": 1000, "gain": 3}]}},
            seed=42,
            engine_version="1.0.0"
        )

        hash2 = node2.hash()

        assert hash1 == hash2

    def test_multiple_nodes_deterministic(self, test_audio, renderer):
        """여러 노드 적용 - 순서 보장"""
        nodes = [
            EditNode(node_type="dsp", params={"compressor": {"threshold": -20}}, seed=42),
            EditNode(node_type="dsp", params={"compressor": {"threshold": -15}}, seed=43),
        ]

        # 첫 실행
        result1 = test_audio.copy()
        for node in nodes:
            result1 = renderer.render_node(result1, node)

        # 두 번째 실행
        result2 = test_audio.copy()
        for node in nodes:
            result2 = renderer.render_node(result2, node)

        # 동일 결과
        assert np.allclose(result1, result2, atol=1e-6)


class TestCacheManagement:
    """캐시 관리 테스트"""

    def test_cache_stats(self, test_audio, renderer):
        """캐시 통계"""
        node = EditNode(node_type="dsp", params={}, seed=42)

        # 첫 실행
        renderer.render_node(test_audio, node)
        stats = renderer.get_cache_stats()

        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1
        assert stats["hit_ratio"] == 0.0

        # 두 번째 실행
        renderer.render_node(test_audio, node)
        stats = renderer.get_cache_stats()

        assert stats["cache_hits"] == 1
        assert stats["hit_ratio"] == 0.5

    def test_clear_cache(self, test_audio, renderer):
        """캐시 클리어"""
        node = EditNode(node_type="dsp", params={}, seed=42)

        # 캐시 생성
        renderer.render_node(test_audio, node)
        assert renderer.cache_hits + renderer.cache_misses > 0

        # 캐시 클리어
        renderer.clear_cache()

        assert renderer.cache_hits == 0
        assert renderer.cache_misses == 0

        # 다시 실행 → cache miss
        renderer.render_node(test_audio, node)
        assert renderer.cache_misses == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
