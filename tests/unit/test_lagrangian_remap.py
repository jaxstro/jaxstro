"""Split and merge on a Lagrangian mesh: plans, exact copies, conservation, the planner."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import lagrangian_remap as lr


def _mesh(n=24, seed=0):
    """A non-uniform mesh with non-trivial fields: faces in xi, masses, two extensive fields
    (one scalar, one with a trailing axis)."""
    rng = np.random.default_rng(seed)
    width = jnp.asarray(rng.uniform(0.5, 1.5, n))
    xi_face = jnp.concatenate([jnp.zeros(1), jnp.cumsum(width)])
    dm = jnp.asarray(rng.uniform(0.2, 2.0, n))
    energy = dm * jnp.asarray(rng.uniform(1.0, 3.0, n))
    species = dm[:, None] * jnp.asarray(rng.dirichlet(np.ones(3), n))
    return xi_face, dm, {"energy": energy, "species": species}


def _scores(xi_face, dm, fields):
    q = fields["energy"] / dm
    split_score = jnp.abs(jnp.gradient(q))
    merge_score = split_score[:-1] + split_score[1:]
    return split_score, merge_score


def test_actions_build_the_expected_plan():
    action = jnp.array([lr.KEEP, lr.SPLIT, lr.MERGE_RIGHT, lr.KEEP, lr.KEEP])
    plan = lr.plan_from_actions(action, capacity=5)
    assert int(plan.n_active) == 5 and bool(plan.is_valid(5))
    # half cells: cell i is (2i, 2i+1); split -> two runs of 1, merge -> one run of 4
    np.testing.assert_array_equal(plan.start, [0, 2, 3, 4, 8])
    np.testing.assert_array_equal(plan.length, [2, 1, 1, 4, 2])
    np.testing.assert_array_equal(plan.is_copy, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(plan.is_split, [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(plan.is_merge, [0, 0, 0, 1, 0])
    # padded capacity: empty slots; too small a capacity: the overflow is reported
    padded = lr.plan_from_actions(action, capacity=7)
    np.testing.assert_array_equal(padded.length[5:], [0, 0])
    assert bool(padded.is_valid(5))
    over = lr.plan_from_actions(jnp.full(5, lr.SPLIT), capacity=6)
    assert int(over.n_active) == 10 and not bool(over.is_valid(5))


def test_a_caller_built_plan_with_a_half_cell_merge():
    """MESA cuts at old half-mass points, so a new cell can be half of one old cell joined
    to the whole of the next. The transfer must accept a plan it did not build."""
    xi_face, dm, fields = _mesh(n=4, seed=1)
    # runs: [cell 0], [inner half of 1], [outer half of 1 + cell 2], [cell 3]
    plan = lr.RemapPlan(
        start=jnp.array([0, 2, 3, 6]),
        length=jnp.array([2, 1, 3, 2]),
        n_active=jnp.array(4),
    )
    assert bool(plan.is_valid(4))
    xi_new, dm_new, new = lr.apply_plan(plan, xi_face, dm, fields)
    np.testing.assert_allclose(
        dm_new, [dm[0], dm[1] / 2, dm[1] / 2 + dm[2], dm[3]], rtol=1e-15
    )
    np.testing.assert_allclose(
        new["energy"][2], fields["energy"][1] / 2 + fields["energy"][2], rtol=1e-15
    )
    np.testing.assert_allclose(xi_new[2], 0.5 * (xi_face[1] + xi_face[2]), rtol=1e-15)
    assert xi_new[3] == xi_face[3] and dm_new[0] == dm[0]


def test_transfer_conserves_and_copies_untouched_cells_bitwise():
    xi_face, dm, fields = _mesh()
    plan = lr.fixed_count_plan(*_scores(xi_face, dm, fields), 0.3, 0.6, max_changes=4)
    n_split = int(jnp.sum(plan.is_split)) // 2
    assert n_split > 0 and n_split == int(jnp.sum(plan.is_merge))  # excited
    slopes = {
        "energy": lr.limited_slopes(fields["energy"] / dm, dm),
        "species": lr.limited_slopes(fields["species"] / dm[:, None], dm, joint=True),
    }
    vslope = lr.limited_slopes(jnp.diff(xi_face) / dm, dm)
    xi_new, dm_new, new = lr.apply_plan(plan, xi_face, dm, fields, slopes, vslope)

    assert dm_new.shape == dm.shape and xi_new.shape == xi_face.shape
    assert jnp.all(jnp.diff(xi_new) > 0.0)
    rel = lambda a, b: float(jnp.max(jnp.abs(a - b) / jnp.abs(b)))  # noqa: E731
    assert rel(jnp.sum(dm_new), jnp.sum(dm)) < 1e-15
    assert xi_new[-1] == xi_face[-1]
    for key in fields:
        assert rel(jnp.sum(new[key], axis=0), jnp.sum(fields[key], axis=0)) < 1e-14
    # positivity is kept by the limited split, and the joint limiter keeps sum-to-one
    assert jnp.all(new["species"] > 0.0) and jnp.all(new["energy"] > 0.0)
    np.testing.assert_allclose(jnp.sum(new["species"], axis=1), dm_new, rtol=1e-14)
    independent = lr.limited_slopes(fields["species"] / dm[:, None], dm)
    _, _, ind = lr.apply_plan(
        plan, xi_face, dm, fields, {"energy": None, "species": independent}
    )
    assert (
        float(jnp.max(jnp.abs(jnp.sum(ind["species"], axis=1) / dm_new - 1.0))) > 1e-6
    )  # control

    copy = plan.is_copy
    p = plan.start[copy] // 2
    assert jnp.array_equal(dm_new[copy], dm[p])
    assert jnp.array_equal(new["species"][copy], fields["species"][p])
    assert jnp.array_equal(xi_new[1:][copy], xi_face[1:][p])
    # control: the slopes do move the children (otherwise the linear rule is untested)
    _, _, flat = lr.apply_plan(plan, xi_face, dm, fields)
    assert rel(flat["energy"], new["energy"]) > 1e-3


def test_split_everything_then_merge_back_is_the_identity():
    xi_face, dm, fields = _mesh(n=10, seed=3)
    slopes = {"energy": lr.limited_slopes(fields["energy"] / dm, dm), "species": None}
    vslope = lr.limited_slopes(jnp.diff(xi_face) / dm, dm)
    fine = lr.plan_from_actions(jnp.full(10, lr.SPLIT), capacity=20)
    xi2, dm2, f2 = lr.apply_plan(fine, xi_face, dm, fields, slopes, vslope)
    back = lr.plan_from_actions(
        jnp.tile(jnp.array([lr.MERGE_RIGHT, lr.KEEP]), 10), capacity=10
    )
    xi3, dm3, f3 = lr.apply_plan(back, xi2, dm2, f2)
    assert jnp.array_equal(dm3, dm)  # halves of a binary float sum back exactly
    assert jnp.array_equal(xi3, xi_face)
    np.testing.assert_allclose(f3["energy"], fields["energy"], rtol=2e-16, atol=0)
    np.testing.assert_allclose(f3["species"], fields["species"], rtol=2e-16, atol=0)


def test_planner_picks_the_extreme_scores_disjointly_and_respects_masks():
    n = 16
    split_score = jnp.zeros(n).at[jnp.array([3, 9, 12])].set(jnp.array([5.0, 7.0, 6.0]))
    merge_score = jnp.full(n - 1, 10.0).at[jnp.array([0, 1, 5, 6])].set(0.1)
    eligible = jnp.ones(n, bool).at[9].set(False)  # a pinned cell
    plan = lr.fixed_count_plan(
        split_score, merge_score, 1.0, 1.0, max_changes=4, split_eligible=eligible
    )
    left = plan.start[plan.is_split][0::2] // 2
    merged = plan.start[plan.is_merge] // 2
    # candidates: splits {3, 12} (9 masked); pairs 0,1 overlap -> 0 wins; 5,6 -> 5 wins
    np.testing.assert_array_equal(np.sort(left), [3, 12])
    np.testing.assert_array_equal(np.sort(merged), [0, 5])
    assert int(plan.n_active) == n
    # a bound on the changes keeps the largest split and the smallest merge
    one = lr.fixed_count_plan(
        split_score, merge_score.at[5].set(0.05), 1.0, 1.0, max_changes=1
    )
    assert int(one.start[one.is_split][0]) // 2 == 9
    assert int(one.start[one.is_merge][0]) // 2 == 5
    # no merge candidate, no change at all
    none = lr.fixed_count_plan(
        split_score, jnp.full(n - 1, 10.0), 1.0, 1.0, max_changes=4
    )
    assert jnp.all(none.is_copy)
    # the cell-ratio limit forces a split of an oversized cell and blocks an oversized merge
    size = jnp.ones(n).at[7].set(3.0).at[10:12].set(1.5)  # merging 10+11 would make 3.0
    scores = jnp.full(n - 1, 0.1).at[4].set(0.0).at[10].set(-1.0)
    ratio = lr.fixed_count_plan(
        jnp.zeros(n), scores, 1.0, 1.0, max_changes=1, size=size, max_ratio=2.5
    )
    assert int(ratio.start[ratio.is_split][0]) // 2 == 7
    assert int(ratio.start[ratio.is_merge][0]) // 2 == 4


def test_plan_and_transfer_trace_under_jit_and_vmap():
    xi_face, dm, fields = _mesh(n=12, seed=5)

    @jax.jit
    def regrid(xi_face, dm, energy):
        q = energy / dm
        s = jnp.abs(jnp.gradient(q))
        plan = lr.fixed_count_plan(s, s[:-1] + s[1:], 0.3, 0.6, max_changes=3)
        return lr.apply_plan(plan, xi_face, dm, energy, lr.limited_slopes(q, dm))

    eager = regrid(xi_face, dm, fields["energy"])
    batched = jax.vmap(regrid)(
        jnp.stack([xi_face, xi_face]),
        jnp.stack([dm, dm]),
        jnp.stack([fields["energy"]] * 2),
    )
    for a, b in zip(eager, batched):
        assert jnp.array_equal(a, b[1])


@pytest.mark.parametrize("bad", [0, 7])
def test_planner_refuses_an_impossible_change_bound(bad):
    with pytest.raises(ValueError):
        lr.fixed_count_plan(jnp.zeros(12), jnp.zeros(11), 1.0, 1.0, max_changes=bad)
