from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from river import stream
from river.checks import check_estimator
from river.cluster import DDStream
from river.cluster.ddstream import _DENSE, _SPARSE, _TRANSITIONAL, DDStreamGrid


def build_ddstream(**kwargs):
    params = dict(
        grid_width=1.0,
        decaying_factor=0.998,
        dense_threshold=2.5,
        sparse_threshold=0.8,
        n_grids=500,
    )
    params.update(kwargs)
    return DDStream(**params)


def test_check_estimator():
    check_estimator(DDStream())


@pytest.mark.parametrize(
    "params",
    [
        {"grid_width": 0},
        {"grid_width": -1},
        {"decaying_factor": 0},
        {"decaying_factor": 1},
        {"decaying_factor": 1.5},
        {"dense_threshold": 1},
        {"dense_threshold": 0.5},
        {"sparse_threshold": 0},
        {"sparse_threshold": 1},
        {"boundary_margin": -0.1},
        {"boundary_margin": 0.5},
        {"boundary_margin": 1.0},
        {"n_grids": 2},
        {"gap": 0},
        {"gap": -5},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        DDStream(**params)


def test_grid_mapping():
    ddstream = build_ddstream(grid_width=0.5)

    assert ddstream._grid_of({0: 0.7, 1: -0.2}) == ((0, 1), (1, -1))
    assert ddstream._grid_of({0: 0.7, 1: 0.2}) == ((0, 1),)
    assert ddstream._grid_of({1: 0.2, 0: 0.7}) == ((0, 1),)
    assert ddstream._grid_of({0: 0.2, 1: 0.2}) == ()
    assert ddstream._grid_of({0: 0.2}) == ()
    assert ddstream._grid_of({0: 1.0}) == ((0, 2),)


def test_shift_keeps_keys_sorted_and_sparse():
    ddstream = build_ddstream()

    assert ddstream._shift(((0, 1), (2, 1)), 1, 3) == ((0, 1), (1, 3), (2, 1))
    assert ddstream._shift(((0, 1), (2, 1)), 3, 3) == ((0, 1), (2, 1), (3, 3))
    assert ddstream._shift(((0, 1), (2, 1)), 0, 5) == ((0, 5), (2, 1))
    assert ddstream._shift(((0, 1), (2, 1)), 2, 0) == ((0, 1),)
    assert ddstream._shift(((0, 1),), 1, 0) == ((0, 1),)


def test_neighbours():
    ddstream = build_ddstream()
    ddstream._features = [0, 1]

    assert set(ddstream._neighbours(((0, 1), (1, 1)))) == {
        ((1, 1),),
        ((0, 2), (1, 1)),
        ((0, 1),),
        ((0, 1), (1, 2)),
    }
    assert set(ddstream._neighbours(())) == {((0, -1),), ((0, 1),), ((1, -1),), ((1, 1),)}


def test_density_update_follows_equation_5():
    lam = 0.998
    ddstream = build_ddstream(decaying_factor=lam, boundary_margin=0)

    for _ in range(3):
        ddstream.learn_one({0: 0.5, 1: 0.5})
    for _ in range(5):
        ddstream.learn_one({0: 9.5, 1: 9.5})
    ddstream.learn_one({0: 0.5, 1: 0.5})

    grid = ddstream.grids[()]
    expected = sum(lam ** (8 - arrival) for arrival in (0, 1, 2, 8))

    assert grid.last_update == 8
    assert grid.density == pytest.approx(expected)


def test_density_never_exceeds_bound():
    lam = 0.99
    ddstream = build_ddstream(decaying_factor=lam)

    for _ in range(500):
        ddstream.learn_one({0: 0.5})

    assert ddstream.grids[()].density <= 1 / (1 - lam)


def test_gap_follows_equation_6():
    ddstream = build_ddstream(n_grids=100, decaying_factor=0.998)

    delta_0 = math.log(0.8 / 2.5) / math.log(0.998)
    delta_1 = math.log((100 - 2.5) / (100 - 0.8)) / math.log(0.998)

    assert ddstream._gap == int(min(delta_0, delta_1))
    assert DDStream(n_grids=10**9)._gap == 1
    assert build_ddstream(gap=7)._gap == 7


def seed_grid(ddstream, key, density, last_update=0):
    grid = DDStreamGrid(last_update)
    grid.density = density
    ddstream._grids[key] = grid
    return grid


def test_grids_are_classified_by_the_density_thresholds():
    ddstream = build_ddstream(n_grids=500, decaying_factor=0.998, gap=1000)
    ddstream._features = [0]

    densities = [3.0, 2.6, 1.5, 0.9, 0.7]
    for coordinate, density in enumerate(densities, start=1):
        seed_grid(ddstream, ((0, coordinate),), density)
    ddstream._inspect()

    assert ddstream._dense_density == pytest.approx(2.5)
    assert ddstream._sparse_density == pytest.approx(0.8)
    assert [ddstream.grids[((0, i),)].attribute for i in range(1, 6)] == [
        _DENSE,
        _DENSE,
        _TRANSITIONAL,
        _TRANSITIONAL,
        _SPARSE,
    ]


def test_boundary_margin_is_relative_to_grid_width():
    assert build_ddstream(grid_width=0.5, boundary_margin=0.1)._margin == pytest.approx(0.05)
    assert build_ddstream(grid_width=4.0, boundary_margin=0.25)._margin == pytest.approx(1.0)


def test_interior_records_are_left_alone():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.5})

    assert sorted(ddstream.grids) == [(), ((0, 1),)]


def test_boundary_record_joins_the_denser_grid():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.02})

    assert sorted(ddstream.grids) == [()]
    assert ddstream.grids[()].density == pytest.approx(
        sum(0.998 ** (20 - arrival) for arrival in range(21))
    )


def test_boundary_record_stays_put_when_the_neighbour_is_denser():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 1.5})
    ddstream.learn_one({0: 1.02})

    assert sorted(ddstream.grids) == [((0, 1),)]


def test_boundary_record_stays_put_when_the_neighbour_is_unknown():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    ddstream.learn_one({0: 1.02})

    assert sorted(ddstream.grids) == [((0, 1),)]


def test_boundary_record_falls_back_on_the_latest_grid_when_densities_tie():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000, decaying_factor=0.998)

    ddstream.learn_one({0: 1.5})
    ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.02})

    assert ddstream.grids[()].last_update == 2
    assert ddstream.grids[((0, 1),)].last_update == 0


def test_a_record_away_from_every_face_is_not_a_boundary_record():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.25})

    assert sorted(ddstream.grids) == [(), ((0, 1),)]


def test_a_record_only_moves_when_the_two_centres_are_close_enough():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.12})

    assert sorted(ddstream.grids) == [(), ((0, 1),)]


def test_only_the_nearby_dimensions_are_considered():
    ddstream = build_ddstream(boundary_margin=0.2, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 0.5, 1: 0.5})
    ddstream.learn_one({0: 1.02, 1: 1.02})

    assert sorted(ddstream.grids) == [(), ((0, 1), (1, 1))]


def test_dcq_means_is_disabled_by_a_zero_margin():
    ddstream = build_ddstream(boundary_margin=0, gap=1000)

    for _ in range(20):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.0})

    assert sorted(ddstream.grids) == [(), ((0, 1),)]


def test_dcq_means_is_skipped_on_inspection_steps():
    ddstream = build_ddstream(boundary_margin=0.2, gap=5)

    for _ in range(5):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 1.02})

    assert sorted(ddstream.grids) == [(), ((0, 1),)]

    ddstream.learn_one({0: 1.02})

    assert ddstream.grids[()].last_update == 6


def test_dcq_means_recovers_a_cluster_split_by_the_grid():
    def run(boundary_margin):
        ddstream = DDStream(
            grid_width=1.0,
            boundary_margin=boundary_margin,
            decaying_factor=0.99,
            dense_threshold=3.0,
            sparse_threshold=0.8,
            n_grids=4,
            gap=10,
        )
        for i in range(600):
            ddstream.learn_one({0: 0.99 if i % 2 else 1.01})
        return ddstream

    split = run(0)
    joined = run(0.1)

    assert sorted(split.grids) == [(), ((0, 1),)]
    assert all(grid.attribute == _TRANSITIONAL for grid in split.grids.values())
    assert split.n_clusters == 0

    assert sorted(joined.grids) == [((0, 1),)]
    assert joined.grids[((0, 1),)].attribute == _DENSE
    assert joined.n_clusters == 1


def test_dcq_means_recovers_blobs_that_straddle_the_grid():
    rng = np.random.RandomState(7)
    centres = np.array([[0.20, 0.20], [0.60, 0.20], [0.20, 0.60], [0.60, 0.60]])
    labels = rng.randint(0, 4, 4000)
    points = centres[labels] + rng.randn(4000, 2) * 0.012

    def run(boundary_margin):
        ddstream = DDStream(
            grid_width=0.2,
            boundary_margin=boundary_margin,
            decaying_factor=0.999,
            dense_threshold=3.0,
            sparse_threshold=0.8,
            n_grids=25,
            gap=20,
        )
        for x, _ in stream.iter_array(points):
            ddstream.learn_one(x)
        predicted = [ddstream.predict_one(x) for x, _ in stream.iter_array(points)]
        return ddstream, adjusted_rand_score(list(labels), predicted)

    split, split_score = run(0)
    joined, joined_score = run(0.2)

    assert split.n_clusters == 0
    assert split_score == pytest.approx(0.0, abs=0.01)

    assert joined.n_clusters == 4
    assert joined_score > 0.7


def test_dcq_means_does_not_hide_a_genuine_second_cluster():
    ddstream = build_ddstream(boundary_margin=0.2, gap=20, decaying_factor=0.99)

    for _ in range(200):
        ddstream.learn_one({0: 0.5})
        ddstream.learn_one({0: 4.02})

    assert sorted(ddstream.grids) == [(), ((0, 4),)]
    assert ddstream.n_clusters == 2


def test_two_clusters_of_adjacent_grids():
    ddstream = build_ddstream()

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
        ddstream.learn_one(x)

    assert ddstream.n_clusters == 2
    assert ddstream.predict_one({0: 1, 1: 1}) == 0
    assert ddstream.predict_one({0: 4, 1: 3}) == 1
    assert set(ddstream.centers) == {0, 1}
    for label, members in ddstream.clusters.items():
        for key in members:
            assert ddstream.grids[key].label == label


def test_no_cluster_before_the_first_gap():
    ddstream = build_ddstream(gap=50)

    for _ in range(50):
        ddstream.learn_one({0: 0.5})

    assert ddstream.n_clusters == 0
    assert ddstream.predict_one({0: 0.5}) == 0

    for _ in range(200):
        ddstream.learn_one({0: 0.5})

    assert ddstream.n_clusters == 1


def test_labels_are_contiguous_from_zero():
    ddstream = build_ddstream(grid_width=0.5)

    for i in range(400):
        for center in (0.25, 5.25, 10.25):
            ddstream.learn_one({0: center + 0.01 * (i % 5)})

    assert sorted(ddstream.clusters) == list(range(ddstream.n_clusters))
    assert ddstream.n_clusters == 3


def test_transitional_grids_extend_a_cluster():
    ddstream = build_ddstream(decaying_factor=0.9, n_grids=8, gap=5, boundary_margin=0)

    for i in range(200):
        ddstream.learn_one({0: 0.5})
        if i % 4 == 0:
            ddstream.learn_one({0: 1.5})

    assert ddstream.grids[()].attribute == _DENSE
    assert ddstream.grids[((0, 1),)].attribute == _TRANSITIONAL
    assert ddstream.n_clusters == 1
    assert ddstream.clusters[0] == {(), ((0, 1),)}


def test_sparse_grids_are_not_clustered():
    ddstream = build_ddstream(decaying_factor=0.9, n_grids=5, gap=5, boundary_margin=0)

    assert ddstream._sparse_density > 1

    for _ in range(50):
        ddstream.learn_one({0: 0.5})
    ddstream.learn_one({0: 50.5})
    for _ in range(4):
        ddstream.learn_one({0: 0.5})

    assert ddstream.grids[((0, 50),)].attribute == _SPARSE
    assert ddstream.grids[((0, 50),)].label is None
    assert ddstream.n_clusters == 1


def test_cluster_splits_when_a_grid_goes_sparse():
    ddstream = build_ddstream(gap=1, decaying_factor=0.9, n_grids=30, boundary_margin=0)

    for _ in range(60):
        for coordinate in (0.5, 1.5, 2.5):
            ddstream.learn_one({0: coordinate})

    assert ddstream.n_clusters == 1
    assert ddstream.clusters[0] == {(), ((0, 1),), ((0, 2),)}

    for _ in range(200):
        ddstream.learn_one({0: 0.5})
        ddstream.learn_one({0: 2.5})

    assert ddstream.n_clusters == 2
    assert ddstream.clusters[0] != ddstream.clusters[1]


def test_clusters_merge_when_a_bridge_becomes_dense():
    ddstream = build_ddstream(gap=1, decaying_factor=0.9, n_grids=30, boundary_margin=0)

    for _ in range(100):
        ddstream.learn_one({0: 0.5})
        ddstream.learn_one({0: 2.5})

    assert ddstream.n_clusters == 2

    for _ in range(100):
        for coordinate in (0.5, 1.5, 2.5):
            ddstream.learn_one({0: coordinate})

    assert ddstream.n_clusters == 1
    assert ddstream.clusters[0] == {(), ((0, 1),), ((0, 2),)}


def test_removal_follows_the_dstream_bound():
    lam, n_grids, sparse_threshold = 0.998, 500, 0.8
    ddstream = build_ddstream(
        decaying_factor=lam, n_grids=n_grids, sparse_threshold=sparse_threshold, gap=1000
    )

    def survives(idle, density_now):
        ddstream._grids = {}
        seed_grid(ddstream, ((0, 1),), density_now / lam**idle)
        ddstream._time = idle
        ddstream._inspect()
        return ((0, 1),) in ddstream.grids

    for idle in (1, 10, 100, 500):
        bound = sparse_threshold * (1 - lam ** (idle + 1)) / (n_grids * (1 - lam))

        assert bound < ddstream._sparse_density
        assert survives(idle, 1.01 * bound)
        assert not survives(idle, 0.99 * bound)


def test_stale_sparse_grids_are_forgotten():
    ddstream = build_ddstream(
        grid_width=0.5, decaying_factor=0.99, n_grids=5, gap=5, boundary_margin=0
    )

    ddstream.learn_one({0: 100.0})
    for _ in range(9):
        ddstream.learn_one({0: 0.1})

    assert ((0, 200),) in ddstream.grids
    assert ddstream.grids[((0, 200),)].attribute == _SPARSE

    for _ in range(5):
        ddstream.learn_one({0: 0.1})

    assert ((0, 200),) not in ddstream.grids


def test_a_sparse_grid_that_just_got_data_is_kept():
    ddstream = build_ddstream(
        grid_width=0.5, decaying_factor=0.99, n_grids=5, gap=5, boundary_margin=0
    )

    for _ in range(20):
        ddstream.learn_one({0: 0.1})
    ddstream.learn_one({0: 100.0})
    for _ in range(4):
        ddstream.learn_one({0: 0.1})

    assert ddstream.grids[((0, 200),)].attribute == _SPARSE
    assert ((0, 200),) in ddstream.grids


def test_a_forgotten_grid_restarts_from_zero_density():
    ddstream = build_ddstream(
        grid_width=0.5, decaying_factor=0.99, n_grids=5, gap=5, boundary_margin=0
    )

    ddstream.learn_one({0: 100.0})
    for _ in range(14):
        ddstream.learn_one({0: 0.1})

    assert ((0, 200),) not in ddstream.grids

    ddstream.learn_one({0: 100.0})

    assert ddstream.grids[((0, 200),)].density == pytest.approx(1.0)


def test_a_tight_gap_does_not_empty_the_grid_list():
    ddstream = build_ddstream(gap=1, decaying_factor=0.99, n_grids=20)

    rng = np.random.RandomState(0)
    for _ in range(2000):
        ddstream.learn_one({0: rng.choice([0.5, 1.5, 8.5])})

    assert sorted(ddstream.grids) == [(), ((0, 1),), ((0, 8),)]
    assert ddstream.n_clusters == 2


def test_grid_list_stays_bounded_under_noise():
    ddstream = build_ddstream(grid_width=0.05, n_grids=400, decaying_factor=0.998)

    X, _ = make_blobs(n_samples=2000, centers=2, cluster_std=0.02, random_state=0)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    rng = np.random.RandomState(0)
    mixed = np.empty((4000, 2))
    mixed[0::2] = X
    mixed[1::2] = rng.rand(2000, 2)

    sizes = []
    for i, (x, _) in enumerate(stream.iter_array(mixed)):
        ddstream.learn_one(x)
        if i > 1000:
            sizes.append(len(ddstream.grids))

    assert max(sizes) < 500


def test_blobs_are_recovered_despite_outliers():
    rng = np.random.RandomState(42)
    X, y = make_blobs(n_samples=3000, centers=4, cluster_std=0.5, random_state=42)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    points = np.vstack([X, rng.rand(500, 2)])
    labels = np.concatenate([y, np.full(500, -1)])
    order = rng.permutation(len(points))
    points, labels = points[order], labels[order]

    ddstream = DDStream(grid_width=0.05, n_grids=400, decaying_factor=0.998)
    for x, _ in stream.iter_array(points):
        ddstream.learn_one(x)

    predicted, truth = [], []
    for (x, _), label in zip(stream.iter_array(points), labels):
        if label != -1:
            predicted.append(ddstream.predict_one(x))
            truth.append(int(label))

    assert ddstream.n_clusters == 4
    assert adjusted_rand_score(truth, predicted) > 0.99


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

    ddstream = DDStream(grid_width=0.06, n_grids=1000, decaying_factor=0.999, gap=20)
    for x, _ in stream.iter_array(points):
        ddstream.learn_one(x)

    predicted = [ddstream.predict_one(x) for x, _ in stream.iter_array(points)]
    sizes = sorted((len(members) for members in ddstream.clusters.values()), reverse=True)

    assert ddstream.n_clusters == 2
    assert sizes[0] > 5 * sizes[1]
    assert adjusted_rand_score(list(labels), predicted) > 0.95


def test_evolving_stream_forgets_old_clusters():
    rng = np.random.RandomState(3)
    segments = [rng.randn(3000, 2) * 0.02 + [0.15 + 0.25 * i, 0.5] for i in range(3)]
    points = np.vstack(segments)

    ddstream = DDStream(grid_width=0.05, n_grids=100, decaying_factor=0.99)
    snapshots = []
    for i, (x, _) in enumerate(stream.iter_array(points)):
        ddstream.learn_one(x)
        if (i + 1) % 3000 == 0:
            snapshots.append([center[0] for center in ddstream.centers.values()])

    assert all(len(centers) == 1 for centers in snapshots)
    assert snapshots[0][0] < snapshots[1][0] < snapshots[2][0]
    assert len(ddstream.grids) < 30


def test_centers_are_density_weighted():
    ddstream = build_ddstream(gap=10, decaying_factor=0.999)

    for _ in range(60):
        ddstream.learn_one({0: 0.5, 1: 0.5})
    for _ in range(60):
        ddstream.learn_one({0: 1.5, 1: 0.5})

    assert ddstream.n_clusters == 1
    center = ddstream.centers[0]
    assert 0.5 < center[0] < 1.5
    assert center[1] == pytest.approx(0.5)


def test_centers_are_only_recomputed_when_needed():
    ddstream = build_ddstream(gap=10)

    for _ in range(100):
        ddstream.learn_one({0: 0.5})

    assert ddstream._stale_centers
    ddstream.centers
    assert not ddstream._stale_centers
    assert ddstream.centers is ddstream.centers


def test_emerging_and_disappearing_features():
    ddstream = build_ddstream(grid_width=0.5, gap=3)

    for _ in range(30):
        ddstream.learn_one({0: 0.6})
    for _ in range(30):
        ddstream.learn_one({0: 0.6, 1: 0.6})
    for _ in range(30):
        ddstream.learn_one({1: 0.6})

    assert ddstream.predict_one({0: 0.6}) >= 0
    assert ddstream.predict_one({}) >= 0
    assert ddstream.predict_one({2: 5.0}) >= 0


def test_predict_one_before_any_learn():
    assert DDStream().predict_one({0: 0.5}) == 0


def test_learn_one_does_not_mutate_input():
    ddstream = build_ddstream()

    for i in range(50):
        x = {0: i * 0.1, 1: 0.5}
        copy = dict(x)
        ddstream.learn_one(x)
        assert x == copy


def test_weighted_records():
    ddstream = build_ddstream(gap=100)

    ddstream.learn_one({0: 0.5}, w=10.0)

    assert ddstream.grids[()].density == pytest.approx(10.0)


def test_pickling_roundtrip():
    ddstream = build_ddstream(grid_width=0.5)

    for i in range(200):
        ddstream.learn_one({0: 0.25 + 0.01 * (i % 4), 1: 0.25})

    restored = pickle.loads(pickle.dumps(ddstream))

    assert restored.n_clusters == ddstream.n_clusters
    assert restored.clusters == ddstream.clusters
    assert restored.predict_one({0: 0.25, 1: 0.25}) == ddstream.predict_one({0: 0.25, 1: 0.25})


def test_clone_starts_fresh():
    ddstream = build_ddstream()

    for _ in range(100):
        ddstream.learn_one({0: 0.5})

    clone = ddstream.clone()

    assert clone.n_clusters == 0
    assert clone.grids == {}
    assert clone._get_params() == ddstream._get_params()
