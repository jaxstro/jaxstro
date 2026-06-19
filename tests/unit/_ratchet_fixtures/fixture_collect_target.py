"""Committed, importable fixture for the ratchet harness's resolve_node_ids test.

Pure stdlib (no jax import) so ``pytest --collect-only`` resolves it quickly in a
subprocess. ``test_real_target`` is a node id that MUST resolve; the harness test also
asks for a sibling ``test_does_not_exist`` node id that must NOT resolve.
"""


def test_real_target():
    assert True
