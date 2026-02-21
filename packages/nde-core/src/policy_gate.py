"""
Policy Gate: QC-0 고객사 정책 검증
"""
from typing import Dict, Any, Tuple, Optional
from .edit_graph import EditNode


class PolicyViolationError(Exception):
    """정책 위반 에러"""
    pass


class PolicyGate:
    """QC-0: 고객사 정책 검증"""

    def __init__(self, policy: Dict[str, Any]):
        """
        Args:
            policy: policy.v1.schema.json 형식의 정책 딕셔너리
        """
        self.policy = policy
        self.allowed_ops = policy.get("allowed_operations", {})
        self.audit_level = policy.get("audit_level", "hash_only")

    def check(self, edit_node: EditNode) -> Tuple[bool, Optional[str]]:
        """
        편집 노드가 정책 위반인지 확인

        Returns:
            (passed: bool, reason: Optional[str])
        """
        # Enhancement only 모드
        if self.allowed_ops.get("enhancement_only", False):
            if edit_node.node_type not in ["dsp", "export", "split", "align", "mix"]:
                return False, f"Policy violation: enhancement_only mode, got {edit_node.node_type}"

        # Style transfer 검사
        if edit_node.node_type == "cover":
            if not self.allowed_ops.get("cover_generation_allowed", False):
                return False, "Policy violation: cover generation not allowed"

        # Inpaint 검사
        if edit_node.node_type == "repaint":
            if not self.allowed_ops.get("inpaint_allowed", False):
                return False, "Policy violation: inpaint not allowed"

        # Time stretch 검사
        if edit_node.node_type == "dsp":
            params = edit_node.params
            if "time_stretch" in params:
                if not self.allowed_ops.get("time_stretch_allowed", False):
                    return False, "Policy violation: time stretch not allowed"

        # Pitch shift 검사
        if edit_node.node_type == "dsp":
            params = edit_node.params
            if "pitch_shift" in params:
                if not self.allowed_ops.get("pitch_shift_allowed", False):
                    return False, "Policy violation: pitch shift not allowed"

        return True, None

    def check_strict(self, edit_node: EditNode) -> None:
        """
        엄격 검사 (예외 발생)

        Raises:
            PolicyViolationError: 정책 위반 시
        """
        passed, reason = self.check(edit_node)
        if not passed:
            raise PolicyViolationError(reason)

    def get_audit_requirements(self) -> Dict[str, bool]:
        """
        감사 요구사항 반환

        Returns:
            {
                "save_graph": bool,
                "save_intermediate_cache": bool,
                "save_source_hash_only": bool
            }
        """
        return {
            "save_graph": self.audit_level in ["full_graph", "graph_with_cache"],
            "save_intermediate_cache": self.audit_level == "graph_with_cache",
            "save_source_hash_only": self.audit_level == "hash_only"
        }


class PolicyLoader:
    """정책 로더 (DB 또는 파일에서)"""

    def __init__(self, default_policy_path: Optional[str] = None):
        self.default_policy_path = default_policy_path
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_policy(self, policy_id: str) -> Dict[str, Any]:
        """
        정책 ID로 정책 로드

        Args:
            policy_id: Policy ULID

        Returns:
            policy.v1.schema.json 형식의 정책 딕셔너리
        """
        # 캐시 확인
        if policy_id in self._cache:
            return self._cache[policy_id]

        # TODO: 실제 구현에서는 DB/파일에서 로드
        # 여기서는 기본 정책 반환
        policy = self._get_default_policy(policy_id)
        self._cache[policy_id] = policy
        return policy

    def _get_default_policy(self, policy_id: str) -> Dict[str, Any]:
        """기본 정책 (enhancement_only)"""
        return {
            "policy_id": policy_id,
            "org_id": "default",
            "policy_name": "enhancement_only",
            "allowed_operations": {
                "enhancement_only": True,
                "inpaint_allowed": False,
                "style_transfer_allowed": False,
                "cover_generation_allowed": False,
                "time_stretch_allowed": True,
                "pitch_shift_allowed": True
            },
            "audit_level": "full_graph",
            "qc_gates": {
                "technical": {
                    "enabled": True,
                    "clipping_ratio_max": 0.001,
                    "lufs_range": {"min": -23, "max": -13},
                    "true_peak_max_db": -1.0,
                    "dc_offset_max": 0.01
                },
                "locality": {
                    "enabled": True,
                    "out_mask_delta_max": 0.05
                },
                "similarity": {
                    "enabled": False
                }
            }
        }

    def create_permissive_policy(self, policy_id: str, org_id: str) -> Dict[str, Any]:
        """전체 기능 허용 정책 (크리에이터용)"""
        return {
            "policy_id": policy_id,
            "org_id": org_id,
            "policy_name": "full_creative",
            "allowed_operations": {
                "enhancement_only": False,
                "inpaint_allowed": True,
                "style_transfer_allowed": True,
                "cover_generation_allowed": True,
                "time_stretch_allowed": True,
                "pitch_shift_allowed": True
            },
            "audit_level": "graph_with_cache",
            "qc_gates": {
                "technical": {
                    "enabled": True,
                    "clipping_ratio_max": 0.005,
                    "lufs_range": {"min": -30, "max": -10},
                    "true_peak_max_db": -0.5,
                    "dc_offset_max": 0.02
                },
                "locality": {
                    "enabled": True,
                    "out_mask_delta_max": 0.15
                },
                "similarity": {
                    "enabled": True,
                    "similarity_threshold": 0.95,
                    "reference_library_path": "/data/reference_library"
                }
            }
        }
