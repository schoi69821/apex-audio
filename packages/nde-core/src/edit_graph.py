"""
Edit Graph: CRDT-style append-only 편집 그래프
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import ulid
import hashlib
import json
import time


@dataclass
class EditNode:
    """CRDT-style append-only 편집 노드"""

    node_type: str  # dsp, repaint, cover, split, align, mix, export
    params: Dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: str(ulid.new()))
    mask: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    engine_version: str = "1.0.0"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    author: str = "system"

    def hash(self) -> str:
        """재현성을 위한 결정적 해시 (seed, params, engine_version 기반)"""
        content = {
            "node_type": self.node_type,
            "params": self._normalize_params(self.params),
            "seed": self.seed,
            "engine_version": self.engine_version
        }
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _normalize_params(params: Dict[str, Any]) -> List[Tuple]:
        """파라미터를 정규화 (순서 독립적)"""
        if not params:
            return []
        return sorted(params.items(), key=lambda x: str(x[0]))

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


class EditGraph:
    """CRDT Edit Graph - append-only, mergeable"""

    def __init__(self, source_hash: str, graph_id: Optional[str] = None):
        self.graph_id = graph_id or str(ulid.new())
        self.source_hash = source_hash
        self.nodes: List[EditNode] = []
        self.created_at = int(time.time() * 1000)
        self.merge_metadata: Dict[str, Any] = {"conflicts": []}

    def append_node(self, node: EditNode) -> str:
        """노드 추가 (append-only)"""
        self.nodes.append(node)
        return node.node_id

    def graph_hash(self) -> str:
        """전체 그래프의 재현성 해시"""
        if not self.nodes:
            return hashlib.sha256(self.source_hash.encode()).hexdigest()

        node_hashes = [n.hash() for n in self.nodes]
        combined = f"{self.source_hash}:{'|'.join(node_hashes)}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def merge(self, other: 'EditGraph') -> 'EditGraph':
        """
        CRDT 병합 - 동일 segment에 대한 충돌 해결

        병합 규칙:
        1. 동일 node_id → 무시 (idempotent)
        2. 서로 다른 노드 → 모두 포함, timestamp 순 정렬
        3. 동일 segment 편집 → last-write-wins (timestamp 기준)
        """
        if self.source_hash != other.source_hash:
            raise ValueError(f"Cannot merge graphs with different sources: {self.source_hash} != {other.source_hash}")

        merged = EditGraph(self.source_hash, graph_id=str(ulid.new()))

        # 모든 노드 수집 (중복 제거)
        node_map = {}
        for node in self.nodes + other.nodes:
            existing = node_map.get(node.node_id)
            if existing is None:
                node_map[node.node_id] = node
            elif existing.timestamp < node.timestamp:
                # Last-write-wins
                merged.merge_metadata["conflicts"].append({
                    "node_id": node.node_id,
                    "resolution": "last-write-wins",
                    "kept_timestamp": node.timestamp,
                    "discarded_timestamp": existing.timestamp
                })
                node_map[node.node_id] = node

        # Timestamp 순 정렬
        merged.nodes = sorted(node_map.values(), key=lambda n: n.timestamp)

        return merged

    def export_patch(self) -> Dict[str, Any]:
        """
        Git patch 스타일 export (원본 없이 전달 가능)

        이 형식으로 전달하면:
        - 원본 오디오 유출 없이
        - Edit Graph만으로 재현 가능
        - 고객사는 자신의 원본 + patch로 로컬 렌더
        """
        return {
            "graph_id": self.graph_id,
            "graph_hash": self.graph_hash(),
            "source_hash": self.source_hash,
            "created_at": self.created_at,
            "nodes": [n.to_dict() for n in self.nodes],
            "merge_metadata": self.merge_metadata
        }

    @classmethod
    def from_patch(cls, patch: Dict[str, Any]) -> 'EditGraph':
        """Patch로부터 EditGraph 복원"""
        graph = cls(
            source_hash=patch["source_hash"],
            graph_id=patch["graph_id"]
        )
        graph.created_at = patch["created_at"]
        graph.merge_metadata = patch.get("merge_metadata", {"conflicts": []})

        for node_dict in patch["nodes"]:
            node = EditNode(**node_dict)
            graph.nodes.append(node)

        return graph

    def get_node(self, node_id: str) -> Optional[EditNode]:
        """노드 ID로 조회"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """그래프 검증"""
        # 1. Source hash 검증
        if not self.source_hash or len(self.source_hash) != 64:
            return False, "Invalid source_hash"

        # 2. 노드 순서 검증 (timestamp 순)
        for i in range(1, len(self.nodes)):
            if self.nodes[i].timestamp < self.nodes[i-1].timestamp:
                return False, f"Nodes not in timestamp order: {self.nodes[i].node_id}"

        # 3. Node ID 중복 검증
        node_ids = [n.node_id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            return False, "Duplicate node_ids found"

        return True, None
