from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from river import stream
from river.checks import check_estimator
from river.cluster import DStream
from river.cluster.dstream import _DENSE, _SPARSE, _TRANSITIONAL


def build_dstream(**kwargs):
    params = dict(
        grid_width=1.0,
        decaying_factor=0.998,
        dense_threshold=2.5,
        sparse_threshold=0.8,
        n_grids=500,
    )
    params.update(kwargs)
    return DStream(**params)


def test_check_estimator():
    check_estimator(DStream())


@pytest.mark.parametrize(
    "params",
    [
        {"grid_width": 0},
        {"grid_width": -1},
        {"decaying_factor": 0},
        {"decaying_factor": 1},
        {"decaying_factor": 1.5},
        {"dense_threshold": 1},
        {"sparse_threshold": 0},
        {"sparse_threshold": 1},
        {"beta": 0},
        {"n_grids": 3},
        {"gap": 0},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        DStream(**params)


def test_grid_mapping():
    dstream = build_dstream(grid_width=0.5)

    assert dstream._grid_of({0: 0.7, 1: -0.2}) == ((0, 1), (1, -1))
    assert dstream._grid_of({0: 0.7, 1: 0.2}) == ((0, 1),)
    assert dstream._grid_of({1: 0.2, 0: 0.7}) == ((0, 1),)
    assert dstream._grid_of({0: 0.2, 1: 0.2}) == ()
    assert dstream._grid_of({0: 0.2}) == ()
    assert dstream._grid_of({0: 1.0}) == ((0, 2),)


def test_density_update_matches_proposition_3_1():
    lam = 0.998
    dstream = build_dstream(decaying_factor=lam)

    for _ in range(3):
        dstream.learn_one({0: 0.5, 1: 0.5})
    for _ in range(5):
        dstream.learn_one({0: 9.5, 1: 9.5})
    dstream.learn_one({0: 0.5, 1: 0.5})

    grid = dstream.grids[()]
    expected = 0.0
    for arrival in (0, 1, 2, 8):
        expected += lam ** (8 - arrival)

    assert grid.last_update == 8
    assert grid.density == pytest.approx(expected)


def test_density_never_exceeds_bound():
    lam = 0.99
    dstream = build_dstream(decaying_factor=lam)

    for _ in range(500):
        dstream.learn_one({0: 0.5})

    assert dstream.grids[()].density <= 1 / (1 - lam)


def test_gap_follows_equation_26():
    dstream = build_dstream(n_grids=100, decaying_factor=0.998)

    delta_0 = math.log(0.8 / 2.5) / math.log(0.998)
    delta_1 = math.log((100 - 2.5) / (100 - 0.8)) / math.log(0.998)

    assert dstream._gap == int(min(delta_0, delta_1))
    assert DStream(n_grids=10**9)._gap == 1
    assert build_dstream(gap=7)._gap == 7


def test_density_thresholds():
    dstream = build_dstream(n_grids=500, decaying_factor=0.998)

    assert dstream._dense_density == pytest.approx(2.5)
    assert dstream._sparse_density == pytest.approx(0.8)
    assert dstream._attribute(3.0) == _DENSE
    assert dstream._attribute(2.6) == _DENSE
    assert dstream._attribute(1.5) == _TRANSITIONAL
    assert dstream._attribute(0.9) == _TRANSITIONAL
    assert dstream._attribute(0.7) == _SPARSE
    assert dstream._attribute(0.0) == _SPARSE


def test_neighbours():
    dstream = build_dstream()
    dstream._features = {0, 1}

    assert set(dstream._neighbours(((0, 1), (1, 1)))) == {
        ((1, 1),),
        ((0, 2), (1, 1)),
        ((0, 1),),
        ((0, 1), (1, 2)),
    }
    assert set(dstream._neighbours(())) == {((0, -1),), ((0, 1),), ((1, -1),), ((1, 1),)}


def test_inside_and_outside_grids():
    dstream = build_dstream()
    dstream._features = {0}

    line = {((0, -1),), (), ((0, 1),)}
    assert dstream._is_inside((), line)
    assert not dstream._is_inside(((0, 1),), line)
    assert not dstream._is_inside(((0, -1),), line)


def test_two_clusters_of_adjacent_grids():
    dstream = build_dstream()

    X = [
        [1, 0.5],
        [1, 0.625],
        [1, 0.75],
        [1, 1.125],
        [1, 1.5],
        [1, 1.75],
        [4, 1.5],
        [4, 2.25],
        [4, 2.5],
        [4, 3],
        [4, 3.25],
        [4, 3.5],
    ]
    for x, _ in stream.iter_array(X):
        dstream.learn_one(x)

    assert dstream.n_clusters == 2
    assert dstream.clusters[0] == {((0, 1),), ((0, 1), (1, 1))}
    assert dstream.clusters[1] == {((0, 4), (1, 3))}
    assert dstream.predict_one({0: 1, 1: 1}) == 0
    assert dstream.predict_one({0: 4, 1: 3}) == 1
    assert set(dstream.centers) == {0, 1}
    assert dstream.centers[1] == pytest.approx({0: 4.5, 1: 3.5})


def test_labels_are_contiguous_from_zero():
    dstream = build_dstream(grid_width=0.5)

    for i in range(400):
        for center in (0.25, 5.25, 10.25):
            dstream.learn_one({0: center + 0.01 * (i % 5)})

    assert sorted(dstream.clusters) == list(range(dstream.n_clusters))
    for label, members in dstream.clusters.items():
        for key in members:
            assert dstream.grids[key].label == label


def test_no_cluster_before_first_gap():
    dstream = build_dstream(gap=50)

    for _ in range(49):
        dstream.learn_one({0: 0.5})

    assert dstream.n_clusters == 0
    assert dstream.predict_one({0: 0.5}) == 0

    for _ in range(200):
        dstream.learn_one({0: 0.5})

    assert dstream.n_clusters == 1


def test_sparse_grids_are_not_clustered():
    dstream = build_dstream(decaying_factor=0.9, n_grids=5, gap=5)

    assert dstream._sparse_density > 1

    for _ in range(50):
        dstream.learn_one({0: 0.5})
    dstream.learn_one({0: 50.5})
    for _ in range(4):
        dstream.learn_one({0: 0.5})

    assert dstream.grids[((0, 50),)].attribute == _SPARSE
    assert dstream.grids[((0, 50),)].label is None
    assert dstream.n_clusters == 1


def test_density_threshold_function():
    dstream = build_dstream(n_grids=500, decaying_factor=0.998, sparse_threshold=0.8)
    dstream._time = 10

    for last_update in (0, 5, 10):
        expected = 0.8 * (1 - 0.998 ** (10 - last_update + 1)) / (500 * (1 - 0.998))
        assert dstream._threshold(last_update) == pytest.approx(expected)

    assert dstream._threshold(0) > dstream._threshold(5) > dstream._threshold(10)
    assert dstream._threshold(0) < dstream._sparse_density


def test_sporadic_grids_are_marked_then_removed():
    dstream = build_dstream(grid_width=0.5, decaying_factor=0.9, n_grids=30, gap=5)

    dstream.learn_one({0: 100.0})
    for _ in range(15):
        dstream.learn_one({0: 0.1})

    assert ((0, 200),) in dstream.grids
    assert dstream.grids[((0, 200),)].sporadic

    for _ in range(5):
        dstream.learn_one({0: 0.1})

    assert ((0, 200),) not in dstream.grids
    assert dstream._removals[((0, 200),)] == 20


def test_removed_grid_restarts_from_zero_density():
    dstream = build_dstream(grid_width=0.5, decaying_factor=0.9, n_grids=30, gap=5)

    dstream.learn_one({0: 100.0})
    for _ in range(20):
        dstream.learn_one({0: 0.1})

    assert ((0, 200),) not in dstream.grids

    dstream.learn_one({0: 100.0})
    grid = dstream.grids[((0, 200),)]

    assert grid.density == pytest.approx(1.0)
    assert grid.last_removal == 20


def test_sporadic_mark_is_reset_when_grid_gets_data():
    dstream = build_dstream(grid_width=0.5, decaying_factor=0.9, n_grids=30, gap=5)

    dstream.learn_one({0: 100.0})
    for _ in range(15):
        dstream.learn_one({0: 0.1})

    assert dstream.grids[((0, 200),)].sporadic

    for _ in range(5):
        dstream.learn_one({0: 100.0})

    assert ((0, 200),) in dstream.grids
    assert not dstream.grids[((0, 200),)].sporadic


def test_removal_bookkeeping_is_pruned():
    dstream = build_dstream(grid_width=0.01, decaying_factor=0.9, n_grids=30, gap=10)

    visited = 4000
    for i in range(visited):
        dstream.learn_one({0: 0.005})
        dstream.learn_one({0: 1 + i * 0.05})

    assert len(dstream.grids) < 50
    assert len(dstream._removals) < visited // 2
    assert all(
        dstream._time < (1 + dstream.beta) * t + dstream._gap for t in dstream._removals.values()
    )


def test_grid_list_stays_bounded_under_noise():
    dstream = build_dstream(grid_width=0.05, n_grids=400, decaying_factor=0.998)
    noisy = build_dstream(
        grid_width=0.05, n_grids=400, decaying_factor=0.998, sparse_threshold=1e-9
    )

    X, _ = make_blobs(n_samples=2000, centers=2, cluster_std=0.02, random_state=0)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    rng = np.random.RandomState(0)
    noise = rng.rand(2000, 2)
    mixed = np.empty((4000, 2))
    mixed[0::2] = X
    mixed[1::2] = noise

    for x, _ in stream.iter_array(mixed):
        dstream.learn_one(x)
        noisy.learn_one(x)

    assert len(dstream.grids) < len(noisy.grids)


def test_cluster_splits_when_a_grid_goes_sparse():
    dstream = build_dstream(grid_width=1.0, gap=1, decaying_factor=0.9, n_grids=30)

    for _ in range(60):
        for coordinate in (0.5, 1.5, 2.5):
            dstream.learn_one({0: coordinate})

    assert dstream.n_clusters == 1
    assert len(dstream.clusters[0]) == 3

    for _ in range(200):
        dstream.learn_one({0: 0.5})
        dstream.learn_one({0: 2.5})

    assert dstream.n_clusters == 2
    assert dstream.clusters[0] != dstream.clusters[1]


def test_clusters_merge_when_a_bridge_becomes_dense():
    dstream = build_dstream(grid_width=1.0, gap=1, decaying_factor=0.9, n_grids=30)

    for _ in range(100):
        dstream.learn_one({0: 0.5})
        dstream.learn_one({0: 2.5})

    assert dstream.n_clusters == 2

    for _ in range(100):
        for coordinate in (0.5, 1.5, 2.5):
            dstream.learn_one({0: coordinate})

    assert dstream.n_clusters == 1
    assert dstream.clusters[0] == {(), ((0, 1),), ((0, 2),)}


def test_arbitrarily_shaped_clusters():
    rng = np.random.RandomState(0)
    angle = rng.rand(1500) * np.pi
    ring = np.stack([np.cos(angle), np.sin(angle)], axis=1) + rng.randn(1500, 2) * 0.03
    ring = (ring - ring.min(0)) / (ring.max(0) - ring.min(0))
    blob = rng.randn(1500, 2) * 0.02 + [0.5, 1.6]
    points = np.vstack([ring, blob])
    labels = np.concatenate([np.zeros(1500), np.ones(1500)])
    order = rng.permutation(len(points))
    points, labels = points[order], labels[order]

    dstream = DStream(grid_width=0.06, n_grids=1000, decaying_factor=0.999, gap=20)
    for x, _ in stream.iter_array(points):
        dstream.learn_one(x)

    predicted = [dstream.predict_one(x) for x, _ in stream.iter_array(points)]
    sizes = sorted((len(members) for members in dstream.clusters.values()), reverse=True)

    assert dstream.n_clusters == 2
    assert sizes[0] > 5 * sizes[1]
    assert adjusted_rand_score(list(labels), predicted) > 0.95


def test_blobs_are_recovered_despite_outliers():
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=3000, centers=4, cluster_std=0.5, random_state=42)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    points = np.vstack([X, rng.rand(500, 2)])
    labels = np.concatenate([y, np.full(500, -1)])
    order = rng.permutation(len(points))
    points, labels = points[order], labels[order]

    dstream = DStream(grid_width=0.05, n_grids=400, decaying_factor=0.998)
    for x, _ in stream.iter_array(points):
        dstream.learn_one(x)

    predicted, truth = [], []
    for (x, _), label in zip(stream.iter_array(points), labels):
        if label != -1:
            predicted.append(dstream.predict_one(x))
            truth.append(int(label))

    assert dstream.n_clusters == 4
    assert adjusted_rand_score(truth, predicted) > 0.99


def test_evolving_stream_forgets_old_clusters():
    rng = np.random.RandomState(3)
    segments = [rng.randn(3000, 2) * 0.02 + [0.15 + 0.25 * i, 0.5] for i in range(3)]
    points = np.vstack(segments)

    dstream = DStream(grid_width=0.05, n_grids=100, decaying_factor=0.99)
    snapshots = []
    for i, (x, _) in enumerate(stream.iter_array(points)):
        dstream.learn_one(x)
        if (i + 1) % 3000 == 0:
            snapshots.append([c[0] for c in dstream.centers.values()])

    assert all(len(centers) == 1 for centers in snapshots)
    assert snapshots[0][0] < snapshots[1][0] < snapshots[2][0]
    assert len(dstream.grids) < 30


def test_emerging_and_disappearing_features():
    dstream = build_dstream(grid_width=0.5, gap=3)

    for _ in range(30):
        dstream.learn_one({0: 0.6})
    for _ in range(30):
        dstream.learn_one({0: 0.6, 1: 0.6})
    for _ in range(30):
        dstream.learn_one({1: 0.6})

    assert dstream.predict_one({0: 0.6}) >= 0
    assert dstream.predict_one({}) >= 0
    assert dstream.predict_one({2: 5.0}) >= 0


def test_learn_one_does_not_mutate_input():
    dstream = build_dstream()

    for i in range(50):
        x = {0: i * 0.1, 1: 0.5}
        copy = dict(x)
        dstream.learn_one(x)
        assert x == copy


def test_weighted_records():
    dstream = build_dstream(gap=100)

    dstream.learn_one({0: 0.5}, w=10.0)

    assert dstream.grids[()].density == pytest.approx(10.0)


def test_pickling_roundtrip():
    dstream = build_dstream(grid_width=0.5)

    for i in range(200):
        dstream.learn_one({0: 0.25 + 0.01 * (i % 4), 1: 0.25})

    restored = pickle.loads(pickle.dumps(dstream))

    assert restored.n_clusters == dstream.n_clusters
    assert restored.clusters == dstream.clusters
    assert restored.predict_one({0: 0.25, 1: 0.25}) == dstream.predict_one({0: 0.25, 1: 0.25})


def test_clone_starts_fresh():
    dstream = build_dstream()

    for _ in range(100):
        dstream.learn_one({0: 0.5})

    clone = dstream.clone()

    assert clone.n_clusters == 0
    assert clone.grids == {}
    assert clone._get_params() == dstream._get_params()
