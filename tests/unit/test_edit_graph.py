"""
Edit Graph 단위 테스트
"""
import pytest
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "nde-core"))

from src.edit_graph import EditNode, EditGraph


class TestEditNode:
    """EditNode 테스트"""

    def test_create_node_with_defaults(self):
        """기본값으로 노드 생성"""
        node = EditNode(
            node_type="dsp",
            params={"eq": {"bands": [{"freq": 1000, "gain": 3}]}}
        )

        assert node.node_type == "dsp"
        assert node.params["eq"]["bands"][0]["freq"] == 1000
        assert node.engine_version == "1.0.0"
        assert node.node_id is not None
        assert node.timestamp > 0

    def test_node_hash_deterministic(self):
        """동일 파라미터 → 동일 해시"""
        node1 = EditNode(
            node_type="dsp",
            params={"eq": {"gain": 3}},
            seed=42,
            engine_version="1.0.0"
        )

        node2 = EditNode(
            node_type="dsp",
            params={"eq": {"gain": 3}},
            seed=42,
            engine_version="1.0.0"
        )

        assert node1.hash() == node2.hash()

    def test_node_hash_different_params(self):
        """다른 파라미터 → 다른 해시"""
        node1 = EditNode(
            node_type="dsp",
            params={"eq": {"gain": 3}},
            seed=42
        )

        node2 = EditNode(
            node_type="dsp",
            params={"eq": {"gain": 6}},
            seed=42
        )

        assert node1.hash() != node2.hash()

    def test_node_hash_different_seed(self):
        """다른 seed → 다른 해시"""
        node1 = EditNode(node_type="dsp", params={}, seed=42)
        node2 = EditNode(node_type="dsp", params={}, seed=43)

        assert node1.hash() != node2.hash()

    def test_node_hash_different_engine_version(self):
        """다른 engine_version → 다른 해시"""
        node1 = EditNode(node_type="dsp", params={}, engine_version="1.0.0")
        node2 = EditNode(node_type="dsp", params={}, engine_version="1.1.0")

        assert node1.hash() != node2.hash()


class TestEditGraph:
    """EditGraph 테스트"""

    def test_create_empty_graph(self):
        """빈 그래프 생성"""
        graph = EditGraph(source_hash="a" * 64)

        assert graph.source_hash == "a" * 64
        assert len(graph.nodes) == 0
        assert graph.graph_id is not None

    def test_append_node(self):
        """노드 추가"""
        graph = EditGraph(source_hash="a" * 64)

        node = EditNode(node_type="dsp", params={"gain": 3})
        node_id = graph.append_node(node)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].node_id == node_id

    def test_graph_hash_deterministic(self):
        """동일 그래프 → 동일 해시"""
        graph1 = EditGraph(source_hash="a" * 64)
        graph1.append_node(EditNode(node_type="dsp", params={"gain": 3}, seed=42, engine_version="1.0.0"))

        graph2 = EditGraph(source_hash="a" * 64)
        graph2.append_node(EditNode(node_type="dsp", params={"gain": 3}, seed=42, engine_version="1.0.0"))

        assert graph1.graph_hash() == graph2.graph_hash()

    def test_export_and_import_patch(self):
        """Export → Import 왕복"""
        graph = EditGraph(source_hash="a" * 64)
        graph.append_node(EditNode(node_type="dsp", params={"gain": 3}, seed=42))
        graph.append_node(EditNode(node_type="repaint", params={"intensity": 0.5}, seed=43))

        # Export
        patch = graph.export_patch()

        assert patch["source_hash"] == "a" * 64
        assert len(patch["nodes"]) == 2

        # Import
        restored = EditGraph.from_patch(patch)

        assert restored.source_hash == graph.source_hash
        assert len(restored.nodes) == len(graph.nodes)
        assert restored.graph_hash() == graph.graph_hash()

    def test_merge_non_conflicting(self):
        """비충돌 병합"""
        graph_a = EditGraph(source_hash="a" * 64)
        node_a = EditNode(node_type="dsp", params={"gain": 3}, seed=42)
        graph_a.append_node(node_a)

        graph_b = EditGraph(source_hash="a" * 64)
        node_b = EditNode(node_type="repaint", params={"intensity": 0.5}, seed=43)
        graph_b.append_node(node_b)

        # 병합
        merged = graph_a.merge(graph_b)

        assert len(merged.nodes) == 2
        # Timestamp 순 정렬 확인
        assert merged.nodes[0].timestamp <= merged.nodes[1].timestamp

    def test_merge_different_source_raises(self):
        """다른 source_hash 병합 → 예외"""
        graph_a = EditGraph(source_hash="a" * 64)
        graph_b = EditGraph(source_hash="b" * 64)

        with pytest.raises(ValueError, match="Cannot merge graphs with different sources"):
            graph_a.merge(graph_b)

    def test_validate_success(self):
        """정상 그래프 검증"""
        graph = EditGraph(source_hash="a" * 64)
        graph.append_node(EditNode(node_type="dsp", params={}))

        passed, reason = graph.validate()
        assert passed is True
        assert reason is None

    def test_validate_invalid_source_hash(self):
        """비정상 source_hash 검증"""
        graph = EditGraph(source_hash="invalid")

        passed, reason = graph.validate()
        assert passed is False
        assert "Invalid source_hash" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
