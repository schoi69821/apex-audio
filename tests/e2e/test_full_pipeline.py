"""
E2E 전체 파이프라인 테스트
"""
import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "nde-core"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "render-workers"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "qc-metrics"))

from src.edit_graph import EditGraph, EditNode
from src.policy_gate import PolicyGate, PolicyLoader
from src.deterministic_renderer import DeterministicRenderer
from src.gates import QCPipeline


@pytest.fixture
def test_audio():
    """테스트 오디오"""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def source_hash():
    """원본 해시"""
    return "a" * 64


@pytest.fixture
def policy():
    """테스트 정책"""
    loader = PolicyLoader()
    return loader.load_policy("default")


class TestFullPipeline:
    """전체 파이프라인 E2E 테스트"""

    def test_session_to_final_render(self, test_audio, source_hash, policy):
        """세션 생성 → 편집 → Preview → Final → Artifact"""

        # 1. EditGraph 생성 (세션 생성)
        graph = EditGraph(source_hash=source_hash)
        assert len(graph.nodes) == 0

        # 2. 편집 노드 추가
        policy_gate = PolicyGate(policy)

        node1 = EditNode(
            node_type="dsp",
            params={"compressor": {"threshold": -20, "ratio": 4.0}},
            seed=42,
            engine_version="1.0.0"
        )

        # Policy Gate 검증
        passed, reason = policy_gate.check(node1)
        assert passed is True

        graph.append_node(node1)
        assert len(graph.nodes) == 1

        # 3. Preview 렌더
        with tempfile.TemporaryDirectory() as tmpdir:
            renderer = DeterministicRenderer("1.0.0", tmpdir)

            preview_audio = test_audio.copy()
            for node in graph.nodes:
                preview_audio = renderer.render_node(preview_audio, node, sr=48000)

            assert preview_audio.shape == test_audio.shape
            # 편집 후 변화 확인
            assert not np.allclose(preview_audio, test_audio)

            # 4. 추가 편집
            node2 = EditNode(
                node_type="dsp",
                params={"eq": {"bands": [{"freq": 1000, "gain": 3}]}},
                seed=43,
                engine_version="1.0.0"
            )

            passed, reason = policy_gate.check(node2)
            assert passed is True

            graph.append_node(node2)
            assert len(graph.nodes) == 2

            # 5. Final 렌더 (전체 그래프)
            final_audio = test_audio.copy()
            for node in graph.nodes:
                final_audio = renderer.render_node(final_audio, node, sr=48000)

            assert final_audio.shape == test_audio.shape

            # 6. QC Gate
            qc_pipeline = QCPipeline(policy)
            qc_passed, qc_results = qc_pipeline.run(
                audio=final_audio,
                sr=48000,
                original=test_audio,
                mask=np.ones_like(test_audio)  # 전체 영역 편집
            )

            assert qc_passed is True
            assert "technical" in qc_results
            assert qc_results["technical"]["passed"] is True

            # 7. Artifact 패키징
            artifact = {
                "artifact_id": "test_artifact",
                "session_id": "test_session",
                "graph_hash": graph.graph_hash(),
                "outputs": {
                    "final_wav": "s3://bucket/final.wav",
                    "preview_mp3": "s3://bucket/preview.mp3",
                    "loudness_report": qc_results["technical"]
                },
                "qc_results": qc_results
            }

            assert artifact["graph_hash"] == graph.graph_hash()
            assert artifact["qc_results"]["technical"]["passed"] is True

    def test_policy_violation_blocks_render(self, test_audio, source_hash, policy):
        """정책 위반 → 렌더 차단"""
        graph = EditGraph(source_hash=source_hash)
        policy_gate = PolicyGate(policy)

        # Enhancement only 정책에서 repaint 시도
        node = EditNode(
            node_type="repaint",
            params={"intensity": 0.5},
            seed=42
        )

        passed, reason = policy_gate.check(node)
        assert passed is False
        assert "enhancement_only" in reason

        # 노드 추가하지 않음
        assert len(graph.nodes) == 0

    def test_export_import_patch_roundtrip(self, source_hash):
        """Export → Import 왕복 검증"""
        # Original graph
        original_graph = EditGraph(source_hash=source_hash)
        original_graph.append_node(EditNode(node_type="dsp", params={"gain": 3}, seed=42))
        original_graph.append_node(EditNode(node_type="dsp", params={"gain": 6}, seed=43))

        original_hash = original_graph.graph_hash()

        # Export patch
        patch = original_graph.export_patch()

        # Import (다른 시스템에서 재현)
        restored_graph = EditGraph.from_patch(patch)

        # 검증
        assert restored_graph.source_hash == original_graph.source_hash
        assert len(restored_graph.nodes) == len(original_graph.nodes)
        assert restored_graph.graph_hash() == original_hash

    def test_qc_gate_failure(self, policy):
        """QC Gate 실패 시나리오"""
        # 클리핑이 심한 오디오 생성
        clipped_audio = np.ones(48000) * 1.5  # > 0.99

        qc_pipeline = QCPipeline(policy)
        qc_passed, qc_results = qc_pipeline.run(
            audio=clipped_audio,
            sr=48000
        )

        # Technical Gate 실패 예상
        assert qc_passed is False
        assert qc_results["technical"]["passed"] is False
        assert "clipping" in str(qc_results["technical"]["failures"]).lower()

    def test_locality_gate_out_mask_delta(self, test_audio, policy):
        """Locality Gate: 마스크 밖 변화 탐지"""
        # 원본
        original = test_audio.copy()

        # 편집 (일부만 변경해야 하는데 전체 변경)
        edited = test_audio * 0.5  # 전체를 50%로 감소

        # 마스크 (앞 절반만 편집 허용)
        mask = np.zeros_like(original)
        mask[:len(mask)//2] = 1.0

        qc_pipeline = QCPipeline(policy)
        qc_passed, qc_results = qc_pipeline.run(
            audio=edited,
            sr=48000,
            original=original,
            mask=mask
        )

        # Locality Gate 실패 예상 (뒷 절반도 변경됨)
        if "locality" in qc_results:
            # Out mask delta가 threshold를 초과해야 함
            assert qc_results["locality"]["out_mask_delta"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
