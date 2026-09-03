from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from river import stream
from river.checks import check_estimator
from river.cluster import MRStream
from river.cluster.mrstream import MRStreamCell, _squared_gap


def build_mrstream(**kwargs):
    params = dict(max_height=3, decaying_factor=0.9, gap=5)
    params.update(kwargs)
    return MRStream(**params)


def seed_cell(mrstream, parent, key, coordinates, weight, last_update=0):
    cell = MRStreamCell(parent.height + 1, coordinates, last_update)
    cell.weight = weight
    parent.children[key] = cell
    mrstream._n_nodes += 1
    return cell


def cells_of(mrstream):
    found = {}
    stack = [mrstream._root]
    while stack:
        cell = stack.pop()
        found[(cell.height, cell.coordinates)] = cell
        stack.extend(cell.children.values())
    return found


def test_check_estimator():
    check_estimator(MRStream())


@pytest.mark.parametrize(
    "params",
    [
        {"max_height": 0},
        {"max_height": -1},
        {"resolution": 0},
        {"resolution": 9},
        {"bounds": (1.0, 0.0)},
        {"bounds": (1.0, 1.0)},
        {"decaying_factor": 0},
        {"decaying_factor": 1},
        {"decaying_factor": 1.5},
        {"dense_threshold": 1},
        {"dense_threshold": 0.5},
        {"sparse_threshold": 0},
        {"sparse_threshold": 1},
        {"n_cells": 1},
        {"epsilon": -1},
        {"mu": 0},
        {"beta": -1},
        {"gap": 0},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        MRStream(**{"max_height": 5, **params})


def test_max_height_has_a_lower_bound():
    with pytest.raises(ValueError, match="must be at least"):
        MRStream(max_height=1)

    smallest = 0.8 * (1 - 0.998**660) / (1 - 0.998)
    height = math.ceil(math.log(smallest, 4))

    assert MRStream(max_height=height).max_height == height
    with pytest.raises(ValueError):
        MRStream(max_height=height - 1)


def test_gap_follows_theorem_7_1():
    mrstream = MRStream(max_height=5, decaying_factor=0.998)

    assert mrstream._gap == int(math.log(0.8 / 3.0) / math.log(0.998))
    assert build_mrstream(gap=7)._gap == 7
    assert MRStream(max_height=5, decaying_factor=0.998)._gap == 660


def test_weight_thresholds_shrink_with_height():
    mrstream = build_mrstream(decaying_factor=0.9, n_cells=4)

    assert mrstream._dense_weights == pytest.approx([30.0, 7.5, 1.875, 0.46875])
    assert mrstream._sparse_weights == pytest.approx([8.0, 2.0, 0.5, 0.125])

    for height in range(mrstream.max_height + 1):
        assert mrstream._dense_weights[height] > mrstream._sparse_weights[height]


def test_coordinates_bisect_the_space():
    mrstream = build_mrstream(max_height=3)

    assert mrstream._codes({0: 0.7}) == [(0, 5)]
    assert mrstream._codes({0: 0.0}) == [(0, 0)]
    assert mrstream._codes({1: 0.2, 0: 0.7}) == [(0, 5), (1, 1)]

    mrstream.learn_one({0: 0.7})
    cells = cells_of(mrstream)

    assert set(cells) == {(0, ()), (1, ((0, 1),)), (2, ((0, 2),)), (3, ((0, 5),))}


def test_values_outside_the_bounds_are_clamped():
    mrstream = build_mrstream(max_height=3)

    assert mrstream._codes({0: -5.0}) == [(0, 0)]
    assert mrstream._codes({0: 1.0}) == [(0, 7)]
    assert mrstream._codes({0: 12.0}) == [(0, 7)]


def test_custom_bounds():
    mrstream = build_mrstream(max_height=3, bounds=(0.0, 10.0))

    assert mrstream._codes({0: 7.0}) == [(0, 5)]
    assert mrstream._codes({0: 0.7}) == [(0, 0)]

    shifted = build_mrstream(max_height=3, bounds=(-1.0, 1.0))

    assert shifted._codes({0: 0.0}) == [(0, 4)]
    assert shifted._codes({0: -1.0}) == [(0, 0)]


def test_a_record_lands_in_one_cell_per_level():
    mrstream = build_mrstream(max_height=3)

    mrstream.learn_one({0: 0.3, 1: 0.3})

    cells = cells_of(mrstream)

    assert mrstream.n_nodes == 4
    assert sorted(height for height, _ in cells) == [0, 1, 2, 3]
    assert all(cell.weight == pytest.approx(1.0) for cell in cells.values())


def test_weight_update_follows_the_fading_model():
    decay = 0.9
    mrstream = build_mrstream(decaying_factor=decay, gap=1000)

    for _ in range(3):
        mrstream.learn_one({0: 0.1})
    for _ in range(5):
        mrstream.learn_one({0: 0.9})
    mrstream.learn_one({0: 0.1})

    cell = cells_of(mrstream)[(3, ())]
    expected = sum(decay ** (8 - arrival) for arrival in (0, 1, 2, 8))

    assert cell.last_update == 8
    assert cell.weight == pytest.approx(expected)


def test_total_weight_never_exceeds_the_bound():
    decay = 0.9
    mrstream = build_mrstream(decaying_factor=decay)

    for i in range(500):
        mrstream.learn_one({0: (i % 7) / 7})

    assert mrstream._root.weight <= 1 / (1 - decay)
    assert mrstream._root.weight == pytest.approx(1 / (1 - decay), rel=1e-3)


def test_weighted_records():
    mrstream = build_mrstream(gap=1000)

    mrstream.learn_one({0: 0.1}, w=10.0)

    assert mrstream._root.weight == pytest.approx(10.0)


def test_up_prune_collapses_dense_siblings():
    mrstream = MRStream(max_height=2, decaying_factor=0.9, gap=1000)
    corners = [(0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375)]

    for i in range(20):
        first, second = corners[i % 4]
        mrstream.learn_one({0: first, 1: second})

    quarter = mrstream._root.children[()]

    assert mrstream.n_nodes == 6
    assert len(quarter.children) == 4
    assert all(child.weight >= mrstream._dense_weights[2] for child in quarter.children.values())

    for i in range(20, 24):
        first, second = corners[i % 4]
        mrstream.learn_one({0: first, 1: second})

    assert mrstream.n_nodes == 2
    assert quarter.children == {}
    assert quarter.implicit_weight == pytest.approx(quarter.weight)


def test_up_prune_needs_every_sibling():
    mrstream = MRStream(max_height=2, decaying_factor=0.9, gap=1000)

    for _ in range(40):
        mrstream.learn_one({0: 0.125, 1: 0.125})
        mrstream.learn_one({0: 0.375, 1: 0.125})

    quarter = mrstream._root.children[()]

    assert len(quarter.children) == 2
    assert quarter.implicit_weight == 0.0


def test_implicit_weight_seeds_a_recreated_child():
    mrstream = MRStream(max_height=2, decaying_factor=0.9, gap=1000)
    corners = [(0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375)]

    for i in range(24):
        first, second = corners[i % 4]
        mrstream.learn_one({0: first, 1: second})

    quarter = mrstream._root.children[()]
    carried = quarter.implicit_weight

    assert quarter.children == {}

    mrstream.learn_one({0: 0.125, 1: 0.125})
    child = quarter.children[()]

    assert child.weight == pytest.approx(carried / 4 + 1)


def test_down_prune_drops_stale_leaves():
    mrstream = MRStream(max_height=2, decaying_factor=0.9, gap=5)

    mrstream.learn_one({0: 0.9, 1: 0.9})
    for _ in range(9):
        mrstream.learn_one({0: 0.1, 1: 0.1})

    assert (2, ((0, 3), (1, 3))) in cells_of(mrstream)

    for _ in range(6):
        mrstream.learn_one({0: 0.1, 1: 0.1})

    cells = cells_of(mrstream)

    assert (2, ((0, 3), (1, 3))) not in cells
    assert (1, ((0, 1), (1, 1))) not in cells
    assert mrstream.n_nodes == len(cells) == 3


def test_down_prune_threshold_follows_the_paper():
    decay = 0.9
    mrstream = build_mrstream(max_height=3, decaying_factor=decay, gap=1)
    mrstream._features = [0]
    mrstream._fanout = 2

    def survives(idle, weight_now):
        mrstream._root.children.clear()
        mrstream._n_nodes = 1
        seed_cell(mrstream, mrstream._root, (), ((0, 1),), weight_now / decay**idle)
        mrstream._time = idle
        mrstream._sweep(mrstream._root)
        return bool(mrstream._root.children)

    for idle in (1, 5, 20, 100):
        bound = mrstream._sparse_weights[1] * (1 - decay ** (idle + 1))

        assert bound < mrstream._sparse_weights[1]
        assert survives(idle, 1.01 * bound)
        assert not survives(idle, 0.99 * bound)


def test_merge_down_collapses_sparse_siblings():
    mrstream = build_mrstream(max_height=2, decaying_factor=0.9, n_cells=4)
    mrstream._features = [0, 1]
    mrstream._fanout = 4
    root = mrstream._root
    root.weight = 7.2

    for key, coordinates in (
        ((), ()),
        ((0,), ((0, 1),)),
        ((1,), ((1, 1),)),
        ((0, 1), ((0, 1), (1, 1))),
    ):
        seed_cell(mrstream, root, key, coordinates, 1.8)

    assert all(cell.weight <= mrstream._sparse_weights[1] for cell in root.children.values())
    assert root.weight <= mrstream._sparse_weights[0]

    mrstream._sweep(root)

    assert root.children == {}
    assert root.implicit_weight == pytest.approx(7.2)
    assert mrstream.n_nodes == 1


def test_merge_down_leaves_a_mixed_family_alone():
    mrstream = build_mrstream(max_height=2, decaying_factor=0.9, n_cells=4)
    mrstream._features = [0, 1]
    mrstream._fanout = 4
    root = mrstream._root
    root.weight = 7.2

    for key, coordinates, weight in (
        ((), (), 1.8),
        ((0,), ((0, 1),), 1.8),
        ((1,), ((1, 1),), 1.8),
        ((0, 1), ((0, 1), (1, 1)), 3.0),
    ):
        seed_cell(mrstream, root, key, coordinates, weight)

    mrstream._sweep(root)

    assert len(root.children) == 4
    assert root.implicit_weight == 0.0


def test_tree_stays_bounded_under_noise():
    mrstream = build_mrstream(max_height=5, decaying_factor=0.99, gap=None)
    rng = np.random.RandomState(0)

    sizes = []
    for i in range(6000):
        mrstream.learn_one({0: rng.rand(), 1: rng.rand()})
        if i > 2000:
            sizes.append(mrstream.n_nodes)

    assert max(sizes) < 4**5
    assert max(sizes) < 2 * min(sizes)


def test_memory_samples_are_taken_at_every_gap():
    mrstream = build_mrstream(gap=10)

    for _ in range(95):
        mrstream.learn_one({0: 0.1})

    assert mrstream.memory_samples == [mrstream.n_nodes] * 9
    assert len(mrstream.memory_samples) == 95 // 10


def test_memory_samples_follow_a_moving_cluster():
    mrstream = MRStream(max_height=5, decaying_factor=0.99, gap=100)
    rng = np.random.RandomState(1)

    for step in range(6):
        for _ in range(1000):
            mrstream.learn_one({0: 0.1 + 0.15 * step + rng.randn() * 0.01})

    samples = mrstream.memory_samples

    assert len(samples) == 59
    assert max(samples) > min(samples)
    assert mrstream.n_nodes < 40


def test_neighbouring_boxes():
    assert _squared_gap([(0, 2)], [(2, 4)], 4) == 0
    assert _squared_gap([(0, 2)], [(3, 4)], 4) == 1
    assert _squared_gap([(0, 2)], [(0, 2)], 4) == 0
    assert _squared_gap([(0, 2), (0, 2)], [(3, 5), (3, 5)], 4) == 2
    assert _squared_gap([(0, 2)], [(9, 11)], 4) >= 4


def test_two_clusters_of_separated_cells():
    mrstream = build_mrstream(max_height=3)

    for _ in range(20):
        mrstream.learn_one({0: 0.3, 1: 0.3})
        mrstream.learn_one({0: 0.55, 1: 0.3})

    assert mrstream.n_clusters == 2
    assert mrstream.clusters[0] == {(3, ((0, 2), (1, 2)))}
    assert mrstream.clusters[1] == {(3, ((0, 4), (1, 2)))}
    assert mrstream.predict_one({0: 0.3, 1: 0.3}) == 0
    assert mrstream.predict_one({0: 0.55, 1: 0.3}) == 1


def test_touching_cells_join():
    mrstream = build_mrstream(max_height=3)

    for _ in range(20):
        mrstream.learn_one({0: 0.3, 1: 0.3})
        mrstream.learn_one({0: 0.4, 1: 0.3})

    assert mrstream.n_clusters == 1
    assert len(mrstream.clusters[0]) == 2


def test_epsilon_controls_the_reach():
    def run(epsilon):
        mrstream = build_mrstream(max_height=3, epsilon=epsilon)
        for _ in range(20):
            mrstream.learn_one({0: 0.3, 1: 0.3})
            mrstream.learn_one({0: 0.55, 1: 0.3})
        return mrstream.n_clusters

    assert run(1.0) == 2
    assert run(1.5) == 1
    assert run(3.0) == 1


def test_a_coarser_resolution_merges_neighbours():
    mrstream = build_mrstream(max_height=3)

    for _ in range(20):
        mrstream.learn_one({0: 0.3, 1: 0.3})
        mrstream.learn_one({0: 0.55, 1: 0.3})

    assert mrstream.n_clusters == 2

    mrstream.mutate({"resolution": 2})

    assert mrstream.n_clusters == 1
    assert mrstream.clusters[0] == {(2, ((0, 1), (1, 1))), (2, ((0, 2), (1, 1)))}
    assert mrstream.predict_one({0: 0.3, 1: 0.3}) == 0
    assert mrstream.predict_one({0: 0.55, 1: 0.3}) == 0


def test_resolution_is_the_only_mutable_attribute():
    mrstream = build_mrstream()

    assert mrstream._mutable_attributes == {"resolution"}

    mrstream.mutate({"resolution": 2})
    assert mrstream.resolution == 2

    with pytest.raises(ValueError):
        mrstream.mutate({"max_height": 2})


def test_the_frontier_stops_at_leaves():
    mrstream = build_mrstream(max_height=3, resolution=3)

    for _ in range(20):
        mrstream.learn_one({0: 0.3, 1: 0.3})

    frontier = mrstream._frontier()

    assert [(cell.height, cell.coordinates) for cell in frontier] == [(3, ((0, 2), (1, 2)))]

    empty = build_mrstream(max_height=3)

    assert [cell.height for cell in empty._frontier()] == [0]


def test_the_frontier_stops_on_implicit_weight():
    mrstream = MRStream(max_height=2, decaying_factor=0.9, gap=1000)
    corners = [(0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375)]

    for i in range(24):
        first, second = corners[i % 4]
        mrstream.learn_one({0: first, 1: second})

    quarter = mrstream._root.children[()]

    assert quarter.implicit_weight > mrstream._sparse_weights[1]
    assert [(cell.height, cell.coordinates) for cell in mrstream._frontier()] == [(1, ())]


def test_only_dense_cells_start_a_cluster():
    mrstream = build_mrstream(max_height=2, gap=1000, mu=1, beta=0.0)

    for _ in range(40):
        mrstream.learn_one({0: 0.1, 1: 0.1})
    mrstream.learn_one({0: 0.9, 1: 0.9})

    lonely = cells_of(mrstream)[(2, ((0, 3), (1, 3)))]
    weight = lonely.weight * mrstream.decaying_factor ** (mrstream._time - lonely.last_update)

    assert mrstream._sparse_weights[2] < weight < mrstream._dense_weights[2]
    assert mrstream.n_clusters == 1
    assert mrstream.clusters[0] == {(2, ())}


def test_a_transitional_cell_joins_a_cluster_that_reaches_it():
    mrstream = build_mrstream(max_height=2, gap=1000, mu=1, beta=0.0)

    for _ in range(40):
        mrstream.learn_one({0: 0.1, 1: 0.1})
    mrstream.learn_one({0: 0.3, 1: 0.1})

    border = cells_of(mrstream)[(2, ((0, 1),))]
    weight = border.weight * mrstream.decaying_factor ** (mrstream._time - border.last_update)

    assert mrstream._sparse_weights[2] < weight < mrstream._dense_weights[2]
    assert mrstream.n_clusters == 1
    assert mrstream.clusters[0] == {(2, ()), (2, ((0, 1),))}


def test_noise_groups_are_dropped():
    def run(mu, beta):
        mrstream = build_mrstream(max_height=3, mu=mu, beta=beta)
        for _ in range(60):
            mrstream.learn_one({0: 0.3, 1: 0.3})
        for _ in range(4):
            mrstream.learn_one({0: 0.9, 1: 0.9})
        return mrstream

    kept = run(mu=1, beta=0.0)
    dropped = run(mu=2, beta=100.0)

    assert kept.n_clusters == 2
    assert dropped.n_clusters == 0


def test_labels_are_contiguous_from_zero():
    mrstream = build_mrstream(max_height=4)

    for _ in range(30):
        for center in (0.1, 0.4, 0.7):
            mrstream.learn_one({0: center, 1: center})

    assert sorted(mrstream.clusters) == list(range(mrstream.n_clusters))
    assert mrstream.n_clusters == 3


def test_centers_are_weight_weighted():
    mrstream = build_mrstream(max_height=3, gap=1000)

    for _ in range(60):
        mrstream.learn_one({0: 0.3, 1: 0.3})
    for _ in range(20):
        mrstream.learn_one({0: 0.4, 1: 0.3})

    assert mrstream.n_clusters == 1
    center = mrstream.centers[0]

    assert 0.3125 < center[0] < 0.4375
    assert center[1] == pytest.approx(0.3125)


def test_clusters_are_refreshed_at_every_gap():
    mrstream = build_mrstream(max_height=3, gap=10)

    for _ in range(21):
        mrstream.learn_one({0: 0.3, 1: 0.3})

    assert mrstream.n_clusters == 1
    assert mrstream.clusters is mrstream.clusters

    for _ in range(9):
        mrstream.learn_one({0: 0.9, 1: 0.9})

    assert mrstream.n_clusters == 1

    mrstream.learn_one({0: 0.9, 1: 0.9})

    assert mrstream.n_clusters == 2


def test_blobs_are_recovered_despite_outliers():
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=3000, centers=4, cluster_std=0.5, random_state=42)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    points = np.vstack([X, rng.rand(500, 2)])
    labels = np.concatenate([y, np.full(500, -1)])
    order = rng.permutation(len(points))
    points, labels = points[order], labels[order]

    mrstream = MRStream(max_height=5)
    for x, _ in stream.iter_array(points):
        mrstream.learn_one(x)

    predicted, truth = [], []
    for (x, _), label in zip(stream.iter_array(points), labels):
        if label != -1:
            predicted.append(mrstream.predict_one(x))
            truth.append(int(label))

    assert mrstream.n_clusters == 4
    assert adjusted_rand_score(truth, predicted) > 0.99


def test_arbitrarily_shaped_clusters():
    rng = np.random.RandomState(0)
    angle = rng.rand(1500) * np.pi
    ring = np.stack([np.cos(angle), np.sin(angle)], axis=1) + rng.randn(1500, 2) * 0.03
    blob = rng.randn(1500, 2) * 0.03 + [0.0, 1.6]
    points = np.vstack([ring, blob])
    points = (points - points.min(0)) / (points.max(0) - points.min(0))
    labels = np.concatenate([np.zeros(1500), np.ones(1500)])
    order = rng.permutation(len(points))
    points, labels = points[order], labels[order]

    mrstream = MRStream(max_height=6, decaying_factor=0.999)
    for x, _ in stream.iter_array(points):
        mrstream.learn_one(x)

    predicted = [mrstream.predict_one(x) for x, _ in stream.iter_array(points)]
    sizes = sorted((len(members) for members in mrstream.clusters.values()), reverse=True)

    assert mrstream.n_clusters == 2
    assert sizes[0] > 5 * sizes[1]
    assert adjusted_rand_score(list(labels), predicted) > 0.99


def test_evolving_stream_forgets_old_clusters():
    rng = np.random.RandomState(3)
    segments = [rng.randn(3000, 2) * 0.02 + [0.15 + 0.25 * i, 0.5] for i in range(3)]
    points = np.vstack(segments)

    mrstream = MRStream(max_height=5, decaying_factor=0.99)
    snapshots = []
    for i, (x, _) in enumerate(stream.iter_array(points)):
        mrstream.learn_one(x)
        if (i + 1) % 3000 == 0:
            snapshots.append([center[0] for center in mrstream.centers.values()])

    assert all(len(centers) == 1 for centers in snapshots)
    assert snapshots[0][0] < snapshots[1][0] < snapshots[2][0]
    assert mrstream.n_nodes < 40


def test_high_dimensional_stream():
    rng = np.random.RandomState(0)
    points = rng.rand(2000, 34) * 0.01
    points[1::2] += 0.80

    mrstream = MRStream(max_height=5, decaying_factor=0.99)
    for x, _ in stream.iter_array(points):
        mrstream.learn_one(x)

    assert mrstream.n_clusters == 2
    assert mrstream.n_nodes == 2 * mrstream.max_height + 1
    assert mrstream.predict_one({i: 0.805 for i in range(34)}) != mrstream.predict_one(
        {i: 0.005 for i in range(34)}
    )


def test_predict_one_before_any_learn():
    assert MRStream().predict_one({0: 0.5}) == 0


def test_predict_one_falls_back_on_the_nearest_center():
    mrstream = build_mrstream(max_height=3)

    for _ in range(20):
        mrstream.learn_one({0: 0.1, 1: 0.1})
        mrstream.learn_one({0: 0.9, 1: 0.9})

    assert mrstream.n_clusters == 2
    assert mrstream.predict_one({0: 0.55, 1: 0.55}) in {0, 1}
    assert mrstream.predict_one({0: 0.2, 1: 0.2}) == mrstream.predict_one({0: 0.1, 1: 0.1})
    assert mrstream.predict_one({0: 0.8, 1: 0.8}) == mrstream.predict_one({0: 0.9, 1: 0.9})


def test_emerging_and_disappearing_features():
    mrstream = build_mrstream(max_height=3, gap=3)

    for _ in range(30):
        mrstream.learn_one({0: 0.6})
    for _ in range(30):
        mrstream.learn_one({0: 0.6, 1: 0.6})
    for _ in range(30):
        mrstream.learn_one({1: 0.6})

    assert mrstream.predict_one({0: 0.6}) >= 0
    assert mrstream.predict_one({}) >= 0
    assert mrstream.predict_one({2: 5.0}) >= 0


def test_learn_one_does_not_mutate_input():
    mrstream = build_mrstream()

    for i in range(50):
        x = {0: i * 0.01, 1: 0.5}
        copy = dict(x)
        mrstream.learn_one(x)
        assert x == copy


def test_pickling_roundtrip():
    mrstream = build_mrstream(max_height=3)

    for i in range(200):
        mrstream.learn_one({0: 0.3 + 0.01 * (i % 4), 1: 0.3})

    restored = pickle.loads(pickle.dumps(mrstream))

    assert restored.n_nodes == mrstream.n_nodes
    assert restored.n_clusters == mrstream.n_clusters
    assert restored.clusters == mrstream.clusters
    assert restored.predict_one({0: 0.3, 1: 0.3}) == mrstream.predict_one({0: 0.3, 1: 0.3})


def test_clone_starts_fresh():
    mrstream = build_mrstream()

    for _ in range(100):
        mrstream.learn_one({0: 0.3, 1: 0.3})

    clone = mrstream.clone()

    assert clone.n_clusters == 0
    assert clone.n_nodes == 1
    assert clone.memory_samples == []
    assert clone._get_params() == mrstream._get_params()
