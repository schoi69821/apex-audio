"""
Policy Gate 단위 테스트
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "nde-core"))

from src.policy_gate import PolicyGate, PolicyViolationError, PolicyLoader
from src.edit_graph import EditNode


class TestPolicyGate:
    """PolicyGate 테스트"""

    def test_enhancement_only_allows_dsp(self):
        """Enhancement only 모드 - DSP 허용"""
        policy = {
            "allowed_operations": {
                "enhancement_only": True
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="dsp", params={"gain": 3})
        passed, reason = gate.check(node)

        assert passed is True
        assert reason is None

    def test_enhancement_only_blocks_repaint(self):
        """Enhancement only 모드 - Repaint 차단"""
        policy = {
            "allowed_operations": {
                "enhancement_only": True
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="repaint", params={"intensity": 0.5})
        passed, reason = gate.check(node)

        assert passed is False
        assert "enhancement_only" in reason

    def test_inpaint_allowed(self):
        """Inpaint 허용"""
        policy = {
            "allowed_operations": {
                "enhancement_only": False,
                "inpaint_allowed": True
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="repaint", params={"intensity": 0.5})
        passed, reason = gate.check(node)

        assert passed is True

    def test_inpaint_blocked(self):
        """Inpaint 차단"""
        policy = {
            "allowed_operations": {
                "inpaint_allowed": False
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="repaint", params={})
        passed, reason = gate.check(node)

        assert passed is False
        assert "inpaint not allowed" in reason

    def test_cover_generation_blocked(self):
        """Cover 생성 차단"""
        policy = {
            "allowed_operations": {
                "cover_generation_allowed": False
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="cover", params={})
        passed, reason = gate.check(node)

        assert passed is False
        assert "cover generation not allowed" in reason

    def test_time_stretch_blocked(self):
        """Time stretch 차단"""
        policy = {
            "allowed_operations": {
                "time_stretch_allowed": False
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="dsp", params={"time_stretch": 1.2})
        passed, reason = gate.check(node)

        assert passed is False
        assert "time stretch not allowed" in reason

    def test_pitch_shift_blocked(self):
        """Pitch shift 차단"""
        policy = {
            "allowed_operations": {
                "pitch_shift_allowed": False
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="dsp", params={"pitch_shift": 2})
        passed, reason = gate.check(node)

        assert passed is False
        assert "pitch shift not allowed" in reason

    def test_check_strict_raises_on_violation(self):
        """엄격 검사 - 위반 시 예외"""
        policy = {
            "allowed_operations": {
                "enhancement_only": True
            }
        }
        gate = PolicyGate(policy)

        node = EditNode(node_type="cover", params={})

        with pytest.raises(PolicyViolationError):
            gate.check_strict(node)

    def test_audit_requirements_hash_only(self):
        """감사 요구사항 - hash_only"""
        policy = {
            "allowed_operations": {},
            "audit_level": "hash_only"
        }
        gate = PolicyGate(policy)

        reqs = gate.get_audit_requirements()

        assert reqs["save_source_hash_only"] is True
        assert reqs["save_graph"] is False
        assert reqs["save_intermediate_cache"] is False

    def test_audit_requirements_full_graph(self):
        """감사 요구사항 - full_graph"""
        policy = {
            "allowed_operations": {},
            "audit_level": "full_graph"
        }
        gate = PolicyGate(policy)

        reqs = gate.get_audit_requirements()

        assert reqs["save_graph"] is True
        assert reqs["save_intermediate_cache"] is False

    def test_audit_requirements_graph_with_cache(self):
        """감사 요구사항 - graph_with_cache"""
        policy = {
            "allowed_operations": {},
            "audit_level": "graph_with_cache"
        }
        gate = PolicyGate(policy)

        reqs = gate.get_audit_requirements()

        assert reqs["save_graph"] is True
        assert reqs["save_intermediate_cache"] is True


class TestPolicyLoader:
    """PolicyLoader 테스트"""

    def test_load_default_policy(self):
        """기본 정책 로드"""
        loader = PolicyLoader()
        policy = loader.load_policy("default")

        assert policy["policy_id"] == "default"
        assert policy["allowed_operations"]["enhancement_only"] is True

    def test_create_permissive_policy(self):
        """전체 기능 허용 정책 생성"""
        loader = PolicyLoader()
        policy = loader.create_permissive_policy("test_id", "test_org")

        assert policy["policy_id"] == "test_id"
        assert policy["allowed_operations"]["inpaint_allowed"] is True
        assert policy["allowed_operations"]["cover_generation_allowed"] is True

    def test_policy_caching(self):
        """정책 캐싱"""
        loader = PolicyLoader()

        policy1 = loader.load_policy("cached")
        policy2 = loader.load_policy("cached")

        # 동일 객체 (캐시 hit)
        assert policy1 is policy2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
