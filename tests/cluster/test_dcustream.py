from __future__ import annotations

import math
import pickle

import pytest

from river import stream
from river.checks import check_estimator
from river.cluster import DCUStream
from river.cluster.dcustream import DCUStreamGrid

DECAY = 0.5


def build_dcustream(**kwargs):
    params = dict(n_intervals=10, bounds=(0.0, 10.0), decaying_factor=1.0, span=0)
    params.update(kwargs)
    return DCUStream(**params)


def seed_grid(dcustream, key, density, last_update=0):
    grid = DCUStreamGrid(last_update)
    grid.density = density
    dcustream._grids[key] = grid
    return grid


def test_check_estimator():
    check_estimator(DCUStream())


@pytest.mark.parametrize(
    "params",
    [
        {"n_intervals": 0},
        {"n_intervals": -3},
        {"bounds": (1.0, 1.0)},
        {"bounds": (2.0, 1.0)},
        {"bounds": (0.0,)},
        {"bounds": (0.0, 1.0, 2.0)},
        {"decaying_factor": 0.999},
        {"decaying_factor": 0},
        {"decaying_factor": -1},
        {"span": -1},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        build_dcustream(**params)


def test_the_paper_requires_a_decaying_factor_of_at_least_one():
    assert build_dcustream(decaying_factor=1.0).decaying_factor == 1.0
    assert build_dcustream(decaying_factor=3.0)._decay == pytest.approx(0.125)


def test_grid_partition_follows_the_bounds():
    dcustream = build_dcustream(n_intervals=4, bounds=(0.0, 1.0))

    assert dcustream._grid_of({0: 0.0}) == ((0, 0),)
    assert dcustream._grid_of({0: 0.24}) == ((0, 0),)
    assert dcustream._grid_of({0: 0.25}) == ((0, 1),)
    assert dcustream._grid_of({0: 0.75}) == ((0, 3),)
    assert dcustream._grid_of({0: 1.0}) == ((0, 3),)


def test_records_outside_the_data_space_land_on_its_edge():
    dcustream = build_dcustream(n_intervals=4, bounds=(0.0, 1.0))

    assert dcustream._grid_of({0: -100.0}) == ((0, 0),)
    assert dcustream._grid_of({0: 100.0}) == ((0, 3),)


def test_a_grid_spans_every_dimension_of_the_record():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))

    assert dcustream._grid_of({0: 3.5, 1: 7.5}) == ((0, 3), (1, 7))
    assert dcustream._grid_of({1: 7.5, 0: 3.5}) == ((0, 3), (1, 7))
    assert dcustream._grid_of({}) == ()


def test_the_number_of_grids_is_the_product_of_the_intervals():
    dcustream = build_dcustream(n_intervals=10)

    assert dcustream._n_grids() == 1
    dcustream.learn_one({0: 1.0})
    assert dcustream._n_grids() == 10
    dcustream.learn_one({0: 1.0, 1: 2.0})
    assert dcustream._n_grids() == 100
    dcustream.learn_one({2: 3.0})
    assert dcustream._n_grids() == 1000


def test_a_very_wide_record_does_not_overflow_the_grid_count():
    dcustream = build_dcustream(n_intervals=10)
    dcustream._known = set(range(400))

    assert dcustream._n_grids() == math.inf
    assert dcustream.min_pts == 0.0


def test_adjacent_grids_share_a_face():
    dcustream = build_dcustream(n_intervals=10)

    assert set(dcustream._neighbours(((0, 3), (1, 7)))) == {
        ((0, 2), (1, 7)),
        ((0, 4), (1, 7)),
        ((0, 3), (1, 6)),
        ((0, 3), (1, 8)),
    }


def test_a_grid_has_at_most_two_adjacent_grids_per_dimension():
    dcustream = build_dcustream(n_intervals=10)

    assert len(set(dcustream._neighbours(tuple((i, 5) for i in range(5))))) == 10


def test_the_partition_has_no_grids_outside_the_data_space():
    dcustream = build_dcustream(n_intervals=4)

    assert set(dcustream._neighbours(((0, 0),))) == {((0, 1),)}
    assert set(dcustream._neighbours(((0, 3),))) == {((0, 2),)}
    assert set(dcustream._neighbours(((0, 0), (1, 3)))) == {((0, 1), (1, 3)), ((0, 0), (1, 2))}


def test_a_new_record_weighs_its_probability():
    dcustream = build_dcustream()

    dcustream.learn_one({0: 5.0}, w=0.7)

    assert dcustream.grids[((0, 5),)].density == pytest.approx(0.7)
    assert dcustream.grids[((0, 5),)].last_update == 0


def test_density_follows_lemma_2():
    dcustream = build_dcustream()

    for _ in range(3):
        dcustream.learn_one({0: 0.5})
    for _ in range(5):
        dcustream.learn_one({0: 9.5})
    dcustream.learn_one({0: 0.5})

    grid = dcustream.grids[((0, 0),)]
    expected = sum(DECAY ** (8 - arrival) for arrival in (0, 1, 2, 8))

    assert grid.last_update == 8
    assert grid.density == pytest.approx(expected)


def test_probabilities_scale_the_density():
    certain = build_dcustream()
    unsure = build_dcustream()

    for _ in range(50):
        certain.learn_one({0: 0.5}, w=1.0)
        unsure.learn_one({0: 0.5}, w=0.25)

    key = ((0, 0),)

    assert unsure.grids[key].density == pytest.approx(0.25 * certain.grids[key].density)


def test_total_density_is_bounded_by_lemma_3():
    dcustream = build_dcustream()

    for i in range(500):
        dcustream.learn_one({0: 0.5 + i % 7})

    total = sum(dcustream._density(grid) for grid in dcustream.grids.values())

    assert total <= 1 / (2 - 2 ** (1 - dcustream.decaying_factor))


def test_min_pts_follows_definition_6():
    dcustream = build_dcustream(n_intervals=10)
    dcustream._known = {0, 1}

    for time in (0, 1, 10, 60):
        dcustream._time = time
        expected = sum(DECAY**tau for tau in range(time + 1)) / (2 * 100 * (1 - DECAY))

        assert dcustream.min_pts == pytest.approx(expected)


def test_min_pts_climbs_to_its_asymptote():
    dcustream = build_dcustream(n_intervals=10)
    dcustream._known = {0, 1}

    thresholds = []
    for time in (0, 2, 5, 10, 60):
        dcustream._time = time
        thresholds.append(dcustream.min_pts)

    assert thresholds == sorted(thresholds)
    assert thresholds[-1] == pytest.approx(1 / (2 * 100 * (1 - DECAY) ** 2))


def test_min_pts_is_inversely_proportional_to_the_grid_count():
    dcustream = build_dcustream(n_intervals=10)
    dcustream._time = 60

    dcustream._known = {0}
    small_space = dcustream.min_pts
    dcustream._known = {0, 1}

    assert small_space == pytest.approx(10 * dcustream.min_pts)


def test_grids_are_split_by_min_pts():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    for interval, factor in enumerate([2.0, 1.0, 0.99, 0.1]):
        seed_grid(dcustream, ((0, interval),), factor * threshold)
    dcustream._recluster()

    assert [dcustream.grids[((0, i),)].dense for i in range(4)] == [True, True, False, False]


def test_a_grid_on_the_threshold_is_dense():
    dcustream = build_dcustream()

    seed_grid(dcustream, ((0, 0),), dcustream.min_pts)
    dcustream._recluster()

    assert dcustream.grids[((0, 0),)].dense is True
    assert dcustream.n_clusters == 1


def test_low_probability_records_do_not_open_a_cluster():
    certain = build_dcustream(n_intervals=10, bounds=(0.0, 1.0))
    unsure = build_dcustream(n_intervals=10, bounds=(0.0, 1.0))

    for _ in range(50):
        certain.learn_one({0: 0.5}, w=0.95)
        unsure.learn_one({0: 0.5}, w=0.05)

    assert certain.n_clusters == 1
    assert unsure.n_clusters == 0
    assert unsure.grids[((0, 5),)].label is None


def test_the_core_dense_grid_seeds_the_cluster():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    seed_grid(dcustream, ((0, 5),), 3 * threshold)
    seed_grid(dcustream, ((0, 0),), 10 * threshold)
    dcustream._recluster()

    assert dcustream.clusters == {0: {((0, 0),)}, 1: {((0, 5),)}}


def test_a_sparse_neighbour_joins_as_a_boundary_grid():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    seed_grid(dcustream, ((0, 0),), 10 * threshold)
    seed_grid(dcustream, ((0, 1),), 0.5 * threshold)
    dcustream._recluster()

    assert dcustream.clusters == {0: {((0, 0),), ((0, 1),)}}
    assert dcustream.grids[((0, 1),)].dense is False


def test_a_boundary_grid_does_not_extend_the_search():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    seed_grid(dcustream, ((0, 0),), 10 * threshold)
    seed_grid(dcustream, ((0, 1),), 0.5 * threshold)
    seed_grid(dcustream, ((0, 2),), 0.5 * threshold)
    dcustream._recluster()

    assert dcustream.clusters == {0: {((0, 0),), ((0, 1),)}}
    assert dcustream.grids[((0, 2),)].label is None


def test_a_sparse_grid_away_from_any_dense_grid_is_noise():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    seed_grid(dcustream, ((0, 0),), 10 * threshold)
    seed_grid(dcustream, ((0, 9),), 0.5 * threshold)
    dcustream._recluster()

    assert dcustream.clusters == {0: {((0, 0),)}}
    assert dcustream.grids[((0, 9),)].label is None
    assert dcustream.predict_one({0: 9.5}) == 0


def test_a_boundary_grid_goes_to_the_densest_of_two_clusters():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    seed_grid(dcustream, ((0, 0),), 3 * threshold)
    seed_grid(dcustream, ((0, 1),), 0.5 * threshold)
    seed_grid(dcustream, ((0, 2),), 10 * threshold)
    dcustream._recluster()

    assert dcustream.clusters == {0: {((0, 2),), ((0, 1),)}, 1: {((0, 0),)}}


def test_dense_grids_assemble_into_one_cluster():
    dcustream = build_dcustream()
    threshold = dcustream.min_pts

    for interval in range(4):
        seed_grid(dcustream, ((0, interval),), 5 * threshold)
    dcustream._recluster()

    assert dcustream.n_clusters == 1
    assert dcustream.clusters[0] == {((0, i),) for i in range(4)}


def test_clusters_take_an_arbitrary_shape():
    dcustream = build_dcustream(n_intervals=10)
    threshold = dcustream.min_pts

    ring = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for column, row in ring:
        seed_grid(dcustream, ((0, column), (1, row)), 5 * threshold)
    dcustream._recluster()

    assert dcustream.n_clusters == 1
    assert len(dcustream.clusters[0]) == 8
    assert ((0, 1), (1, 1)) not in dcustream.clusters[0]


def test_two_clusters_arriving_side_by_side():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))

    X = [
        [1, 0.5],
        [4, 3.0],
        [1, 0.75],
        [4, 3.25],
        [1, 1.5],
        [4, 3.5],
        [1, 0.6],
        [4, 3.1],
        [1, 1.6],
        [4, 3.4],
        [1, 0.7],
        [4, 3.2],
    ]
    for x, _ in stream.iter_array(X):
        dcustream.learn_one(x)

    assert dcustream.n_clusters == 2
    assert dcustream.predict_one({0: 4, 1: 3}) == 0
    assert dcustream.predict_one({0: 1, 1: 0.5}) == 1
    assert set(dcustream.centers) == {0, 1}
    for label, members in dcustream.clusters.items():
        for key in members:
            assert dcustream.grids[key].label == label


def test_labels_are_contiguous_from_zero():
    dcustream = build_dcustream(n_intervals=100, bounds=(0.0, 10.0))

    for _ in range(20):
        for value in (0.5, 4.5, 8.5):
            dcustream.learn_one({0: value})

    assert dcustream.n_clusters == 3
    assert sorted(dcustream.clusters) == [0, 1, 2]


def test_clusters_merge_when_a_bridge_becomes_dense():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))

    for _ in range(20):
        dcustream.learn_one({0: 0.5})
        dcustream.learn_one({0: 2.5})

    assert dcustream.n_clusters == 2

    for _ in range(20):
        for value in (0.5, 1.5, 2.5):
            dcustream.learn_one({0: value})

    assert dcustream.n_clusters == 1
    assert dcustream.clusters[0] == {((0, 0),), ((0, 1),), ((0, 2),)}


def test_a_cluster_splits_when_its_bridge_goes_sparse():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))

    for _ in range(20):
        for value in (0.5, 1.5, 2.5):
            dcustream.learn_one({0: value})

    assert dcustream.n_clusters == 1

    for _ in range(20):
        dcustream.learn_one({0: 0.5})
        dcustream.learn_one({0: 2.5})

    assert dcustream.n_clusters == 2


def test_a_grid_that_stops_receiving_records_goes_sparse():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))

    dcustream.learn_one({0: 9.5})
    for _ in range(20):
        dcustream.learn_one({0: 0.5})

    assert dcustream.grids[((0, 9),)].label is None
    assert dcustream.n_clusters == 1


def test_no_clusters_before_the_span_has_elapsed():
    dcustream = build_dcustream(span=10)

    for _ in range(9):
        dcustream.learn_one({0: 0.5})

    assert dcustream.n_clusters == 0
    assert dcustream.centers == {}
    assert dcustream.predict_one({0: 0.5}) == 0

    dcustream.learn_one({0: 0.5})

    assert dcustream.n_clusters == 1


def test_centers_are_density_weighted():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))
    threshold = dcustream.min_pts

    seed_grid(dcustream, ((0, 0), (1, 0)), 3 * threshold)
    seed_grid(dcustream, ((0, 1), (1, 0)), threshold)
    dcustream._recluster()

    assert dcustream.n_clusters == 1
    center = dcustream.centers[0]
    assert center[0] == pytest.approx((3 * 0.5 + 1 * 1.5) / 4)
    assert center[1] == pytest.approx(0.5)


def test_reclustering_is_cached_until_the_next_learn():
    dcustream = build_dcustream()

    for _ in range(10):
        dcustream.learn_one({0: 0.5})

    assert not dcustream._clustered
    clusters = dcustream.clusters
    assert dcustream._clustered
    assert dcustream.clusters is clusters

    dcustream.learn_one({0: 0.5})

    assert not dcustream._clustered


def test_predict_one_before_any_learn():
    assert DCUStream().predict_one({0: 0.5}) == 0


def test_emerging_and_disappearing_features():
    dcustream = build_dcustream()

    for _ in range(10):
        dcustream.learn_one({0: 6.0})
    for _ in range(10):
        dcustream.learn_one({0: 6.0, 1: 6.0})
    for _ in range(10):
        dcustream.learn_one({1: 6.0})

    assert dcustream.predict_one({0: 6.0}) >= 0
    assert dcustream.predict_one({}) >= 0
    assert dcustream.predict_one({2: 5.0}) >= 0


def test_learn_one_does_not_mutate_input():
    dcustream = build_dcustream()

    for i in range(50):
        x = {0: i * 0.1, 1: 5.0}
        copy = dict(x)
        dcustream.learn_one(x)
        assert x == copy


def test_pickling_roundtrip():
    dcustream = build_dcustream(n_intervals=10, bounds=(0.0, 10.0))

    for i in range(30):
        dcustream.learn_one({0: 2.5 + 0.1 * (i % 4), 1: 2.5})

    restored = pickle.loads(pickle.dumps(dcustream))

    assert restored.n_clusters == dcustream.n_clusters
    assert restored.clusters == dcustream.clusters
    assert restored.predict_one({0: 2.5, 1: 2.5}) == dcustream.predict_one({0: 2.5, 1: 2.5})


def test_clone_starts_fresh():
    dcustream = build_dcustream()

    for _ in range(30):
        dcustream.learn_one({0: 0.5})

    clone = dcustream.clone()

    assert clone.n_clusters == 0
    assert clone.grids == {}
    assert clone._get_params() == dcustream._get_params()
