from __future__ import annotations

import math
import pickle
import random
import warnings

import pytest

from river import stream
from river.checks import check_estimator
from river.cluster import HPStream
from river.cluster.hpstream import HPStreamFadingCluster

DECAY = 0.5


def build_hpstream(**kwargs):
    params = dict(
        max_clusters=4,
        n_projected_dimensions=2,
        decaying_factor=1.0,
        n_samples_init=0,
        stream_speed=1,
        seed=42,
    )
    params.update(kwargs)
    return HPStream(**params)


def assert_centers_close(left, right):
    assert set(left) == set(right)
    for label, center in left.items():
        assert center == pytest.approx(right[label])


def seed_cluster(hpstream, label, points, dimensions=None, last_update=0, last_seen=0):
    """Put a cluster holding `points` in place, without going through `learn_one`."""
    features = sorted({feature for point in points for feature in point}, key=repr)
    for feature in features:
        if feature not in hpstream._sigma:
            hpstream._register(feature)
    cluster = hpstream._new_cluster(points[0], 1.0)
    for point in points[1:]:
        hpstream._add_point(cluster, point, 1.0)
    cluster.last_update = last_update
    cluster.last_seen = last_seen
    cluster.dimensions = list(features if dimensions is None else dimensions)
    hpstream._clusters[label] = cluster
    return cluster


def three_groups():
    """Three groups told apart by the first two dimensions; the third is noise."""
    return [
        {0: 1.0, 1: 1.0, 2: 0.0},
        {0: 1.2, 1: 0.8, 2: 9.0},
        {0: 0.9, 1: 1.1, 2: 4.0},
        {0: 1.1, 1: 1.2, 2: 6.0},
        {0: 5.0, 1: 5.0, 2: 8.0},
        {0: 5.2, 1: 4.8, 2: 1.0},
        {0: 4.9, 1: 5.1, 2: 5.0},
        {0: 5.1, 1: 5.2, 2: 2.0},
        {0: 9.0, 1: 9.0, 2: 3.0},
        {0: 9.2, 1: 8.8, 2: 7.0},
        {0: 8.9, 1: 9.1, 2: 0.5},
        {0: 9.1, 1: 9.2, 2: 9.5},
    ]


def two_subspace_groups(n=200, seed=42):
    """Two groups, each tight along the two dimensions the other one is noise along."""
    rng = random.Random(seed)
    points = []
    for i in range(n):
        if i % 2 == 0:
            points.append(
                {
                    0: rng.gauss(0, 0.1),
                    1: rng.gauss(0, 0.1),
                    2: rng.uniform(0, 10),
                    3: rng.uniform(0, 10),
                }
            )
        else:
            points.append(
                {
                    0: rng.uniform(0, 10),
                    1: rng.uniform(0, 10),
                    2: rng.gauss(5, 0.1),
                    3: rng.gauss(5, 0.1),
                }
            )
    return points


def test_check_estimator():
    check_estimator(HPStream())


def test_check_estimator_without_the_offline_phase():
    check_estimator(HPStream(n_samples_init=0, max_clusters=3, seed=42))


def test_check_estimator_with_a_radius_threshold():
    check_estimator(HPStream(radius_threshold=0.5, n_samples_init=10, seed=42))


@pytest.mark.parametrize(
    "params",
    [
        {"max_clusters": 0},
        {"max_clusters": -1},
        {"n_projected_dimensions": 0},
        {"n_projected_dimensions": -2},
        {"radius_threshold": -0.1},
        {"spread_radius_factor": 0},
        {"spread_radius_factor": -1},
        {"decaying_factor": 0},
        {"decaying_factor": -0.5},
        {"n_samples_init": -1},
        {"stream_speed": 0},
        {"stream_speed": -10},
        {"normalization_interval": 0},
        {"normalization_interval": -5},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        build_hpstream(**params)


# The fading cluster structure of Definition 2.2.


def test_a_cluster_holds_the_first_two_moments_of_its_points():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}, {0: 5.0}])

    assert cluster.weight == pytest.approx(3.0)
    assert cluster.sum1[0] == pytest.approx(9.0)
    assert cluster.sum2[0] == pytest.approx(35.0)
    assert cluster.center[0] == pytest.approx(3.0)


def test_a_point_counts_for_its_weight():
    hpstream = build_hpstream()
    cluster = hpstream._new_cluster({0: 2.0}, 0.25)

    assert cluster.weight == pytest.approx(0.25)
    assert cluster.sum1[0] == pytest.approx(0.5)
    assert cluster.sum2[0] == pytest.approx(1.0)
    assert cluster.center[0] == pytest.approx(2.0)


def test_clusters_add_up_as_in_observation_2_1():
    hpstream = build_hpstream()
    left = seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}])
    right = seed_cluster(hpstream, 1, [{0: 5.0}, {0: 7.0}])
    both = seed_cluster(hpstream, 2, [{0: 1.0}, {0: 3.0}, {0: 5.0}, {0: 7.0}])

    assert both.weight == pytest.approx(left.weight + right.weight)
    assert both.sum1[0] == pytest.approx(left.sum1[0] + right.sum1[0])
    assert both.sum2[0] == pytest.approx(left.sum2[0] + right.sum2[0])


def test_a_stale_cluster_decays_as_in_observation_2_2():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 2.0}, {0: 4.0}])
    weight, linear, squares = cluster.weight, cluster.sum1[0], cluster.sum2[0]

    hpstream._timestamp = 3
    hpstream._add_point(cluster, {0: 0.0}, 0.0)

    assert cluster.weight == pytest.approx(weight * DECAY**3)
    assert cluster.sum1[0] == pytest.approx(linear * DECAY**3)
    assert cluster.sum2[0] == pytest.approx(squares * DECAY**3)


def test_decay_leaves_the_centroid_alone():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 2.0}, {0: 4.0}])

    hpstream._timestamp = 5
    hpstream._add_point(cluster, {0: 0.0}, 0.0)

    assert cluster.center[0] == pytest.approx(3.0)


def test_the_clock_ticks_once_every_stream_speed_points():
    hpstream = build_hpstream(stream_speed=10)

    for _ in range(9):
        hpstream.learn_one({0: 1.0})
    assert hpstream._timestamp == 0

    hpstream.learn_one({0: 1.0})
    assert hpstream._timestamp == 1

    for _ in range(10):
        hpstream.learn_one({0: 1.0})
    assert hpstream._timestamp == 2


def test_a_faded_point_weighs_less_than_a_recent_one():
    recent = build_hpstream(max_clusters=1, stream_speed=1)
    faded = build_hpstream(max_clusters=1, stream_speed=1)

    recent.learn_one({0: 0.0})
    for _ in range(10):
        recent.learn_one({0: 1.0})
    for _ in range(10):
        faded.learn_one({0: 1.0})
    faded.learn_one({0: 0.0})

    assert recent.clusters[0].center[0] > faded.clusters[0].center[0]


def test_the_radius_of_a_dimension_follows_equation_1():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}, {0: 5.0}])

    assert cluster.radius(0) == pytest.approx(math.sqrt(8 / 3))


def test_the_radius_of_a_solitary_cluster_is_zero():
    hpstream = build_hpstream()
    cluster = hpstream._new_cluster({0: 4.0, 1: -2.0}, 1.0)

    assert cluster.radius(0) == 0.0
    assert cluster.radius(1) == 0.0


# Normalisation.


def test_dimensions_are_divided_by_their_standard_deviation():
    hpstream = build_hpstream(normalization_interval=4)

    for value in (0.0, 2.0, 4.0, 6.0):
        hpstream.learn_one({0: value, 1: 0.5 * value})

    assert hpstream._sigma[0] == pytest.approx(math.sqrt(20 / 3))
    assert hpstream._sigma[1] == pytest.approx(math.sqrt(5 / 3))
    assert hpstream._normalize({0: 10.0}) == pytest.approx({0: 10.0 / math.sqrt(20 / 3)})


def test_an_unseen_dimension_is_left_as_is():
    hpstream = build_hpstream()

    assert hpstream._normalize({"never": 3.0}) == {"never": 3.0}


def test_a_constant_dimension_is_not_rescaled():
    hpstream = build_hpstream(normalization_interval=5)

    for _ in range(10):
        hpstream.learn_one({0: 7.0})

    assert hpstream._sigma[0] == 1.0


def test_rescaling_keeps_the_clusters_in_the_units_of_the_stream():
    hpstream = build_hpstream(normalization_interval=1_000_000)
    seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}])
    for _ in range(4):
        hpstream._observe({0: 0.0})
        hpstream._observe({0: 8.0})

    centers = hpstream.centers
    hpstream._renormalize()

    assert hpstream._sigma[0] != 1.0
    assert_centers_close(hpstream.centers, centers)


def test_rescaling_follows_the_ratio_of_the_deviations():
    hpstream = build_hpstream(normalization_interval=1_000_000)
    cluster = seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}])
    linear, squares = cluster.sum1[0], cluster.sum2[0]
    for value in (0.0, 4.0, 8.0, 12.0):
        hpstream._observe({0: value})

    hpstream._renormalize()
    ratio = 1.0 / hpstream._sigma[0]

    assert cluster.sum1[0] == pytest.approx(linear * ratio)
    assert cluster.sum2[0] == pytest.approx(squares * ratio * ratio)


def test_normalization_puts_dimensions_of_different_scales_on_equal_footing():
    hpstream = build_hpstream(max_clusters=2, n_projected_dimensions=1, n_samples_init=200)
    rng = random.Random(0)

    # Both dimensions separate the two groups just as well, but the second is a thousand times
    # wider in raw units. Without normalisation its radius would always be the larger one.
    for i in range(200):
        group = i % 2
        hpstream.learn_one({0: group + rng.gauss(0, 0.01), 1: 1000.0 * (group + rng.gauss(0, 0.2))})

    assert hpstream.n_clusters == 2
    for dimensions in hpstream.dimensions.values():
        assert dimensions == [0]


# Choosing the projected dimensions, as in Figure 3.


def test_the_tightest_dimensions_are_the_ones_kept():
    hpstream = build_hpstream(n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 0.1, 1: 5.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert hpstream.dimensions == {0: [0]}


def test_the_budget_is_the_number_of_clusters_times_the_projection():
    hpstream = build_hpstream(n_projected_dimensions=2)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}])
    seed_cluster(hpstream, 1, [{0: 9.0, 1: 9.0, 2: 9.0, 3: 9.0}, {0: 8.0, 1: 7.0, 2: 6.0, 3: 5.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert sum(len(dimensions) for dimensions in hpstream.dimensions.values()) == 4


def test_a_tight_cluster_may_keep_more_dimensions_than_a_loose_one():
    hpstream = build_hpstream(n_projected_dimensions=2)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0, 2: 0.0}, {0: 0.01, 1: 0.01, 2: 0.01}])
    seed_cluster(hpstream, 1, [{0: 9.0, 1: 9.0, 2: 9.0}, {0: 0.0, 1: 0.0, 2: 0.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)
    kept = hpstream.dimensions

    assert len(kept[0]) == 3
    assert len(kept[1]) == 1


def test_the_ranking_is_global_and_can_leave_a_cluster_with_nothing():
    """Figure 3 hands the budget out by rank alone, with nothing held back per cluster."""
    hpstream = build_hpstream(n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 0.01, 1: 0.02}])
    seed_cluster(hpstream, 1, [{0: 9.0, 1: 9.0}, {0: 8.0, 1: 5.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert hpstream.dimensions == {0: [0, 1], 1: []}


def test_a_cluster_the_ranking_starves_is_deleted():
    """The `Remove those clusters ... which have zero dimensions` line of Figure 1."""
    hpstream = build_hpstream(n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 0.01, 1: 0.02}])
    seed_cluster(hpstream, 1, [{0: 9.0, 1: 9.0}, {0: 8.0, 1: 5.0}])

    hpstream._update({0: 0.005, 1: 0.01}, 1.0)

    assert 1 not in hpstream.clusters


def test_dimensions_are_listed_tightest_first():
    hpstream = build_hpstream(n_projected_dimensions=3)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0, 2: 0.0}, {0: 3.0, 1: 1.0, 2: 2.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert hpstream.dimensions == {0: [1, 2, 0]}


def test_dimensions_stay_in_rank_order_when_only_some_of_them_are_kept():
    hpstream = build_hpstream(n_projected_dimensions=2)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0, 2: 0.0}, {0: 3.0, 1: 1.0, 2: 2.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert hpstream.dimensions == {0: [1, 2]}


def test_the_projection_cannot_ask_for_more_dimensions_than_there_are():
    hpstream = build_hpstream(n_projected_dimensions=10)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 1.0, 1: 2.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert sorted(hpstream.dimensions[0]) == [0, 1]


def test_the_default_projection_keeps_most_of_the_dimensions():
    hpstream = build_hpstream(n_projected_dimensions=None)
    for feature in range(10):
        hpstream._register(feature)

    assert hpstream._projection_size() == 6


def test_the_incoming_point_is_added_before_the_radii_are_ranked():
    hpstream = build_hpstream(n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}])

    # On its own the cluster has a radius of zero along both dimensions. The incoming point is
    # what separates them, which is the degenerate case Section 3 calls out.
    hpstream._compute_dimensions(hpstream._clusters, {0: 0.1, 1: 5.0}, 1.0)

    assert hpstream.dimensions == {0: [0]}


def test_a_cluster_of_one_point_looks_tight_along_every_dimension():
    """r_j = |p_j - x_j| / 2, which is what lets a fresh cluster take the whole budget."""
    hpstream = build_hpstream(n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}])
    seed_cluster(hpstream, 1, [{0: 5.0, 1: 5.0}, {0: 6.0, 1: 7.0}])

    hpstream._compute_dimensions(hpstream._clusters, {0: 0.01, 1: 0.01}, 1.0)

    assert sorted(hpstream.dimensions[0]) == [0, 1]
    assert hpstream.dimensions[1] == []


# The radius threshold of Section 3, used instead of the rank.


def test_the_radius_threshold_keeps_every_tight_dimension():
    hpstream = build_hpstream(radius_threshold=1.0)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0, 2: 0.0}, {0: 0.5, 1: 1.0, 2: 8.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert sorted(hpstream.dimensions[0]) == [0, 1]


def test_the_radius_threshold_ignores_the_number_of_dimensions_asked_for():
    hpstream = build_hpstream(radius_threshold=1.0, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0, 2: 0.0}, {0: 0.5, 1: 1.0, 2: 8.0}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert len(hpstream.dimensions[0]) == 2


def test_a_cluster_tight_along_nothing_is_dropped_under_a_radius_threshold():
    hpstream = build_hpstream(radius_threshold=0.01)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 0.001, 1: 0.001}])
    seed_cluster(hpstream, 1, [{0: 9.0, 1: 9.0}, {0: 0.0, 1: 0.0}])

    hpstream._update({0: 0.0005, 1: 0.0005}, 1.0)

    assert 1 not in hpstream.clusters


def test_the_number_of_projected_dimensions_moves_with_a_radius_threshold():
    hpstream = build_hpstream(radius_threshold=1.0)
    cluster = seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 0.5, 1: 0.5}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)
    assert len(hpstream.dimensions[0]) == 2

    # The second dimension spreads out, and drops out of the projection on its own.
    for value in (5.0, -5.0, 8.0):
        hpstream._add_point(cluster, {0: 0.25, 1: value}, 1.0)

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)
    assert hpstream.dimensions[0] == [0]


# The projected distance of Figure 2.


def test_the_projected_distance_only_reads_the_dimensions_of_the_cluster():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}], dimensions=[0])

    assert hpstream._projected_distance({0: 3.0, 1: 1000.0}, cluster) == pytest.approx(3.0)


def test_the_projected_distance_is_an_average_over_the_dimensions():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}], dimensions=[0, 1])

    assert hpstream._projected_distance({0: 3.0, 1: 1.0}, cluster) == pytest.approx(2.0)


def test_a_missing_dimension_is_read_as_zero():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 4.0, 1: 4.0}], dimensions=[0, 1])

    assert hpstream._projected_distance({0: 4.0}, cluster) == pytest.approx(2.0)


def test_a_cluster_without_dimensions_is_infinitely_far_away():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 0.0}], dimensions=[])

    assert hpstream._projected_distance({0: 0.0}, cluster) == math.inf


def test_the_closest_cluster_is_the_one_the_point_joins():
    hpstream = build_hpstream()
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 2.0}], dimensions=[0])
    seed_cluster(hpstream, 1, [{0: 10.0}, {0: 12.0}], dimensions=[0])

    assert hpstream._closest({0: 9.0}, hpstream._clusters)[0] == 1
    assert hpstream._closest({0: 3.0}, hpstream._clusters)[0] == 0


# The limiting radius of Figure 4.


def test_the_limiting_radius_scales_the_average_radius():
    hpstream = build_hpstream(spread_radius_factor=2.0)
    cluster = seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}, {0: 5.0}], dimensions=[0])

    assert hpstream._limiting_radius(cluster) == pytest.approx(2 * math.sqrt(8 / 3))


def test_the_limiting_radius_averages_over_the_projected_dimensions_only():
    hpstream = build_hpstream()
    points = [{0: 1.0, 1: 0.0}, {0: 3.0, 1: 0.0}, {0: 5.0, 1: 0.0}]
    both = seed_cluster(hpstream, 0, points, dimensions=[0, 1])
    one = seed_cluster(hpstream, 1, points, dimensions=[0])

    assert hpstream._limiting_radius(both) == pytest.approx(
        hpstream._limiting_radius(one) / math.sqrt(2)
    )


def test_a_point_inside_the_boundary_joins_the_cluster():
    hpstream = build_hpstream(n_projected_dimensions=1)
    cluster = seed_cluster(hpstream, 0, [{0: 0.0}, {0: 2.0}, {0: 4.0}])

    hpstream._update({0: 2.5}, 1.0)

    assert hpstream.n_clusters == 1
    assert cluster.weight == pytest.approx(4.0)


def test_a_point_outside_the_boundary_starts_a_cluster_of_its_own():
    hpstream = build_hpstream(n_projected_dimensions=1)
    cluster = seed_cluster(hpstream, 0, [{0: 0.0}, {0: 2.0}, {0: 4.0}])

    hpstream._update({0: 100.0}, 1.0)

    assert hpstream.n_clusters == 2
    assert cluster.weight == pytest.approx(3.0)
    assert hpstream.clusters[1].center[0] == pytest.approx(100.0)


# The degenerate case of Figure 4, kept as the paper writes it. These tests pin down what the
# algorithm does wrong, so that a change of behaviour shows up as a failing test rather than
# as a silently different clustering.


def test_a_cluster_with_no_spread_has_a_boundary_of_zero():
    hpstream = build_hpstream()
    solitary = seed_cluster(hpstream, 0, [{0: 3.0}], dimensions=[0])

    assert hpstream._spread(solitary) == 0.0
    assert hpstream._limiting_radius(solitary) == 0.0


def test_raising_the_spread_factor_does_not_rescue_a_cluster_with_no_spread():
    for factor in (2.0, 10.0, 1000.0):
        hpstream = build_hpstream(spread_radius_factor=factor)
        solitary = seed_cluster(hpstream, 0, [{0: 3.0}], dimensions=[0])

        assert hpstream._limiting_radius(solitary) == 0.0


def test_a_cluster_created_online_can_never_take_a_second_point():
    hpstream = build_hpstream(max_clusters=8, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 1.0}, {0: 2.0}])

    hpstream._update({0: 50.0}, 1.0)
    born = max(hpstream.clusters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(1, 21):
            hpstream._update({0: 50.0 + 0.001 * i}, 1.0)

    assert hpstream.clusters[born].weight == pytest.approx(1.0)


def test_the_model_collapses_once_every_slot_holds_a_single_point():
    hpstream = build_hpstream(max_clusters=3, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 1.0}, {0: 2.0}])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(200):
            hpstream._update({0: 50.0 + 0.001 * i}, 1.0)

    assert hpstream.n_degenerate_clusters == hpstream.n_clusters
    assert all(cluster.weight == pytest.approx(1.0) for cluster in hpstream.clusters.values())


def test_the_papers_spread_factor_lets_a_subspace_stream_collapse():
    fragile = HPStream(
        max_clusters=2,
        n_projected_dimensions=2,
        spread_radius_factor=2,
        n_samples_init=100,
        stream_speed=200,
        seed=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for point in two_subspace_groups():
            fragile.learn_one(point)

    assert fragile.n_spawned_clusters > 0
    assert fragile.n_degenerate_clusters == fragile.n_clusters


def test_a_wide_enough_boundary_keeps_the_model_out_of_the_degenerate_state():
    """The same stream, with `spread_radius_factor` raised from the paper's 2."""
    safe = HPStream(
        max_clusters=2,
        n_projected_dimensions=2,
        spread_radius_factor=3,
        n_samples_init=100,
        stream_speed=200,
        seed=0,
    )
    for point in two_subspace_groups():
        safe.learn_one(point)

    assert safe.n_spawned_clusters == 0
    assert safe.n_degenerate_clusters == 0
    assert min(cluster.weight for cluster in safe.clusters.values()) > 1


def test_a_warning_is_raised_once_the_model_starts_turning_points_away():
    hpstream = build_hpstream(max_clusters=3, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 1.0}, {0: 2.0}])

    with pytest.warns(UserWarning, match="spread_radius_factor"):
        for i in range(200):
            hpstream._update({0: 50.0 + 0.001 * i}, 1.0)


def test_the_warning_is_only_raised_once():
    hpstream = build_hpstream(max_clusters=3, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 1.0}, {0: 2.0}])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(500):
            hpstream._update({0: 50.0 + 0.001 * i}, 1.0)

    assert len(caught) == 1


def test_no_warning_on_a_stream_that_stays_healthy():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hpstream = HPStream(max_clusters=3, n_projected_dimensions=2, n_samples_init=12, seed=0)
        for point in three_groups():
            hpstream.learn_one(point)

    assert hpstream.n_degenerate_clusters == 0


def test_a_warning_is_raised_when_the_offline_phase_produces_a_dead_cluster():
    # Each group holds identical points, so the k-means hands back clusters with no spread.
    with pytest.warns(UserWarning, match="n_samples_init"):
        hpstream = build_hpstream(max_clusters=2, n_samples_init=10)
        for i in range(10):
            hpstream.learn_one({0: float(i % 2), 1: float(i % 2)})

    assert hpstream.n_degenerate_clusters > 0


def test_the_starvation_rule_can_delete_a_healthy_cluster():
    """Three healthy clusters, a budget of k*l = 6 and k*d = 9 candidates: one is squeezed out."""
    hpstream = build_hpstream(
        max_clusters=3,
        n_projected_dimensions=2,
        spread_radius_factor=3,
        decaying_factor=0.5,
        stream_speed=200,
        n_samples_init=100,
    )
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(100):
            group = rng.randrange(3)
            hpstream.learn_one({j: 5 * group + rng.gauss(0, 0.5) for j in range(3)})
        assert hpstream.n_clusters == 3
        assert hpstream.n_degenerate_clusters == 0

        for _ in range(200):
            group = rng.randrange(3)
            hpstream.learn_one({j: 5 * group + rng.gauss(0, 0.5) for j in range(3)})

    # One of the three groups lost its cluster, and the points that used to belong to it now
    # open clusters of their own that Figure 4 will not let grow.
    assert hpstream.n_spawned_clusters > 0
    assert hpstream.n_degenerate_clusters > 0


def test_a_generous_projection_budget_makes_starvation_impossible():
    """k*l > (k-1)*d leaves a slot for every cluster however the ranking falls."""
    hpstream = build_hpstream(
        max_clusters=3,
        n_projected_dimensions=3,
        spread_radius_factor=3,
        decaying_factor=0.5,
        stream_speed=200,
        n_samples_init=100,
    )
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(300):
            group = rng.randrange(3)
            hpstream.learn_one({j: 5 * group + rng.gauss(0, 0.5) for j in range(3)})

    assert hpstream.n_clusters == 3
    assert hpstream.n_spawned_clusters == 0
    assert hpstream.n_degenerate_clusters == 0


def test_a_radius_threshold_cannot_starve_a_cluster_by_competition():
    """The threshold is read per cluster, so no cluster can take another one's dimensions."""
    hpstream = build_hpstream(radius_threshold=1.0)
    seed_cluster(hpstream, 0, [{0: 0.0, 1: 0.0}, {0: 0.01, 1: 0.02}])
    seed_cluster(hpstream, 1, [{0: 9.0, 1: 9.0}, {0: 9.5, 1: 9.4}])

    hpstream._compute_dimensions(hpstream._clusters, None, 0.0)

    assert sorted(hpstream.dimensions[0]) == [0, 1]
    assert sorted(hpstream.dimensions[1]) == [0, 1]


def test_the_diagnostics_start_at_zero():
    hpstream = build_hpstream()

    assert hpstream.n_degenerate_clusters == 0
    assert hpstream.n_spawned_clusters == 0


# Holding at most `max_clusters` clusters.


def test_a_new_cluster_evicts_the_least_recently_written_one():
    hpstream = build_hpstream(max_clusters=2, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 1.0}], last_seen=0)
    seed_cluster(hpstream, 1, [{0: 10.0}, {0: 11.0}], last_seen=5)

    hpstream._n_samples_seen = 6
    hpstream._update({0: 100.0}, 1.0)

    assert set(hpstream.clusters) == {0, 1}
    assert hpstream.clusters[0].center[0] == pytest.approx(100.0)
    assert hpstream.clusters[1].center[0] == pytest.approx(10.5)


def test_the_label_of_an_evicted_cluster_is_handed_to_the_newcomer():
    hpstream = build_hpstream(max_clusters=2, n_projected_dimensions=1)
    seed_cluster(hpstream, 0, [{0: 0.0}, {0: 1.0}], last_seen=0)
    seed_cluster(hpstream, 1, [{0: 10.0}, {0: 11.0}], last_seen=5)

    hpstream._n_samples_seen = 6
    hpstream._update({0: 100.0}, 1.0)

    assert sorted(hpstream.clusters) == [0, 1]


def test_the_number_of_clusters_never_goes_past_the_maximum():
    hpstream = build_hpstream(max_clusters=3, n_samples_init=0, n_projected_dimensions=1)
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(500):
            hpstream.learn_one({0: rng.uniform(0, 100), 1: rng.uniform(0, 100)})
            assert hpstream.n_clusters <= 3


def test_labels_stay_within_the_maximum():
    hpstream = build_hpstream(max_clusters=3, n_samples_init=0)
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(500):
            hpstream.learn_one({0: rng.uniform(0, 100), 1: rng.uniform(0, 100)})
            assert all(0 <= label < 3 for label in hpstream.clusters)


# The offline initialisation.


def test_nothing_is_clustered_before_the_sample_is_in():
    hpstream = build_hpstream(n_samples_init=10)

    for _ in range(9):
        hpstream.learn_one({0: 1.0, 1: 1.0})

    assert hpstream.n_clusters == 0
    assert hpstream.centers == {}
    assert hpstream.dimensions == {}
    assert hpstream.predict_one({0: 1.0, 1: 1.0}) == 0
    assert len(hpstream._init_buffer) == 9


def test_the_clusters_are_seeded_once_the_sample_is_in():
    hpstream = build_hpstream(n_samples_init=10, max_clusters=2)
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(10):
            group = float(i % 2)
            hpstream.learn_one({0: group + rng.gauss(0, 0.01), 1: group + rng.gauss(0, 0.01)})

    assert hpstream.n_clusters == 2
    assert hpstream._init_buffer == []
    assert hpstream._initialized


def test_the_offline_phase_separates_well_separated_groups():
    hpstream = build_hpstream(n_samples_init=60, max_clusters=3, n_projected_dimensions=2)
    rng = random.Random(0)

    for i in range(60):
        center = (0.0, 20.0, 40.0)[i % 3]
        hpstream.learn_one({0: center + rng.gauss(0, 0.5), 1: center + rng.gauss(0, 0.5)})

    centers = sorted(center[0] for center in hpstream.centers.values())

    assert hpstream.n_clusters == 3
    assert centers == pytest.approx([0.0, 20.0, 40.0], abs=1.0)


def test_the_offline_phase_is_reproducible():
    left = build_hpstream(n_samples_init=60, seed=7)
    right = build_hpstream(n_samples_init=60, seed=7)
    rng = random.Random(0)
    points = [{0: rng.uniform(0, 10), 1: rng.uniform(0, 10)} for _ in range(60)]

    for point in points:
        left.learn_one(point)
        right.learn_one(point)

    assert_centers_close(left.centers, right.centers)
    assert left.dimensions == right.dimensions


def test_the_offline_refinement_stops_even_when_it_does_not_converge():
    """`repeated iteratively until the procedure converges` is capped, as k-means always is."""
    from river.cluster.hpstream import _INIT_ITERATIONS

    assert _INIT_ITERATIONS > 0

    hpstream = build_hpstream(n_samples_init=400, max_clusters=10, n_projected_dimensions=3)
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(400):
            hpstream.learn_one({j: rng.uniform(0, 1) for j in range(6)})

    assert hpstream._initialized


def test_a_sample_smaller_than_the_number_of_clusters_is_handled():
    hpstream = build_hpstream(n_samples_init=2, max_clusters=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hpstream.learn_one({0: 0.0})
        hpstream.learn_one({0: 5.0})

    assert hpstream.n_clusters <= 2
    assert hpstream._initialized


def test_the_offline_phase_survives_a_sample_of_identical_points():
    hpstream = build_hpstream(n_samples_init=20, max_clusters=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):
            hpstream.learn_one({0: 3.0, 1: 3.0})

    assert hpstream._initialized
    assert hpstream.n_clusters >= 1


def test_the_offline_phase_can_be_skipped():
    hpstream = build_hpstream(n_samples_init=0)

    assert hpstream._initialized

    hpstream.learn_one({0: 1.0, 1: 1.0})

    assert hpstream.n_clusters == 1


def test_the_seeds_of_the_offline_kmeans_are_spread_out():
    hpstream = build_hpstream(n_samples_init=0)
    points = [({0: 0.0}, 1.0)] * 20 + [({0: 100.0}, 1.0)] * 20

    seeds = hpstream._seeds(points, 2)

    assert sorted(point[0] for point, _ in seeds) == [0.0, 100.0]


def test_seeding_asks_for_no_more_seeds_than_there_are_distinct_points():
    hpstream = build_hpstream(n_samples_init=0)
    points = [({0: 1.0}, 1.0)] * 5

    assert len(hpstream._seeds(points, 4)) == 4


# Projected clustering end to end.


def test_the_noise_dimension_is_thrown_away():
    hpstream = HPStream(max_clusters=3, n_projected_dimensions=2, n_samples_init=12, seed=0)

    for point in three_groups():
        hpstream.learn_one(point)

    assert hpstream.n_clusters == 3
    for dimensions in hpstream.dimensions.values():
        assert sorted(dimensions) == [0, 1]


def test_clusters_living_in_different_subspaces_are_found():
    hpstream = HPStream(
        max_clusters=2,
        n_projected_dimensions=2,
        spread_radius_factor=3,
        n_samples_init=100,
        stream_speed=200,
        seed=0,
    )

    for point in two_subspace_groups():
        hpstream.learn_one(point)

    assert hpstream.n_clusters == 2
    assert {tuple(sorted(dimensions)) for dimensions in hpstream.dimensions.values()} == {
        (0, 1),
        (2, 3),
    }


def test_points_are_assigned_to_the_cluster_of_their_own_subspace():
    hpstream = HPStream(
        max_clusters=2,
        n_projected_dimensions=2,
        spread_radius_factor=3,
        n_samples_init=100,
        stream_speed=200,
        seed=0,
    )
    points = two_subspace_groups()

    for point in points:
        hpstream.learn_one(point)

    tight_on_first = {
        label for label, dimensions in hpstream.dimensions.items() if sorted(dimensions) == [0, 1]
    }
    assert len(tight_on_first) == 1
    label = tight_on_first.pop()

    assert all(hpstream.predict_one(point) == label for point in points[::2])
    assert all(hpstream.predict_one(point) != label for point in points[1::2])


def test_the_centers_are_in_the_units_the_points_arrived_in():
    hpstream = build_hpstream(max_clusters=1, n_samples_init=50, normalization_interval=10)
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(50):
            hpstream.learn_one({0: 1000.0 + rng.gauss(0, 1.0)})

    assert hpstream.centers[0][0] == pytest.approx(1000.0, abs=5.0)


def test_a_cluster_follows_a_group_that_drifts():
    hpstream = build_hpstream(
        max_clusters=1,
        n_samples_init=0,
        stream_speed=1,
        decaying_factor=1.0,
        spread_radius_factor=100,
    )
    rng = random.Random(0)

    # A half-life of one record leaves the cluster holding little more than the last point, so
    # it has next to no spread and Figure 4 keeps turning points away — the documented failure
    # mode. What the test is after is that the centroid follows the group all the same.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(50):
            hpstream.learn_one({0: rng.gauss(0, 0.1)})
        assert hpstream.centers[0][0] == pytest.approx(0.0, abs=0.5)

        for _ in range(50):
            hpstream.learn_one({0: rng.gauss(10, 0.1)})

    assert hpstream.centers[0][0] == pytest.approx(10.0, abs=0.5)


# River conventions.


def test_predict_one_before_any_learn():
    assert HPStream().predict_one({0: 1.0}) == 0


def test_predict_one_does_not_mutate_the_model():
    hpstream = build_hpstream(n_samples_init=20)
    rng = random.Random(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(20):
            hpstream.learn_one({0: rng.uniform(0, 3), 1: rng.uniform(0, 2)})

    before = pickle.dumps(hpstream)
    hpstream.predict_one({0: 1.0, 1: 1.0})

    assert pickle.dumps(hpstream) == before


def test_learn_one_does_not_mutate_its_input():
    hpstream = build_hpstream(n_samples_init=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(50):
            x = {0: i * 0.1, 1: 5.0}
            copy = dict(x)
            hpstream.learn_one(x)
            assert x == copy


def test_the_model_does_not_hold_on_to_the_dict_it_was_given():
    hpstream = build_hpstream(n_samples_init=5)

    x = {0: 1.0, 1: 2.0}
    hpstream.learn_one(x)
    before = pickle.dumps(hpstream)
    x[0] = 1e9
    x.pop(1)

    assert pickle.dumps(hpstream) == before


def test_the_order_of_the_features_does_not_matter():
    # A realistic decay: `build_hpstream` halves the weight of a point at every record, which
    # leaves every cluster holding its last point and no spread at all.
    settings = dict(
        n_samples_init=100,
        max_clusters=3,
        n_projected_dimensions=3,
        spread_radius_factor=3,
        decaying_factor=0.5,
        stream_speed=200,
    )
    straight = build_hpstream(**settings)
    shuffled = build_hpstream(**settings)
    rng = random.Random(0)

    for _ in range(200):
        group = rng.randrange(3)
        x = {j: 5 * group + rng.gauss(0, 0.5) if j < 2 else rng.uniform(0, 15) for j in range(4)}
        keys = list(x)
        rng.shuffle(keys)
        straight.learn_one(x)
        shuffled.learn_one({key: x[key] for key in keys})

    assert_centers_close(straight.centers, shuffled.centers)
    # The ranking of Figure 3 breaks exact ties by the order the dimensions were first seen,
    # so the dimensions are compared as sets rather than in the order they are listed.
    assert {label: set(kept) for label, kept in straight.dimensions.items()} == {
        label: set(kept) for label, kept in shuffled.dimensions.items()
    }


def test_emerging_and_disappearing_features():
    hpstream = build_hpstream(n_samples_init=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(10):
            hpstream.learn_one({0: 6.0})
        for _ in range(10):
            hpstream.learn_one({0: 6.0, 1: 6.0})
        for _ in range(10):
            hpstream.learn_one({1: 6.0})

    assert hpstream.predict_one({0: 6.0}) >= 0
    assert hpstream.predict_one({}) >= 0
    assert hpstream.predict_one({2: 5.0}) >= 0
    for cluster in hpstream.clusters.values():
        assert set(cluster.sum1) == {0, 1}
        assert set(cluster.sum2) == {0, 1}


def test_an_empty_record_leaves_no_cluster_behind():
    hpstream = build_hpstream(n_samples_init=0)

    hpstream.learn_one({})

    assert hpstream.n_clusters == 0


def test_a_weightless_point_opens_no_cluster():
    hpstream = build_hpstream(n_samples_init=0)

    hpstream.learn_one({0: 1.0}, w=0.0)

    assert hpstream.n_clusters == 0
    assert hpstream._moments[0][0] == 1


def test_a_weighted_point_counts_for_more():
    hpstream = build_hpstream()

    light = hpstream._new_cluster({0: 0.0}, 1.0)
    hpstream._add_point(light, {0: 10.0}, 1.0)
    heavy = hpstream._new_cluster({0: 0.0}, 1.0)
    hpstream._add_point(heavy, {0: 10.0}, 4.0)

    assert light.center[0] == pytest.approx(5.0)
    assert heavy.center[0] == pytest.approx(8.0)


def test_string_feature_names_are_handled():
    hpstream = build_hpstream(n_samples_init=10, max_clusters=2)
    rng = random.Random(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(50):
            hpstream.learn_one({"height": rng.gauss(0, 1), "width": rng.gauss(10, 1)})

    assert hpstream.n_clusters >= 1
    for dimensions in hpstream.dimensions.values():
        assert set(dimensions) <= {"height", "width"}


def test_pickling_roundtrip():
    hpstream = build_hpstream(n_samples_init=30, max_clusters=3)
    rng = random.Random(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(100):
            hpstream.learn_one({0: rng.uniform(0, 5), 1: rng.uniform(0, 5)})

    restored = pickle.loads(pickle.dumps(hpstream))

    assert restored.n_clusters == hpstream.n_clusters
    assert_centers_close(restored.centers, hpstream.centers)
    assert restored.dimensions == hpstream.dimensions
    assert restored.predict_one({0: 2.0, 1: 2.0}) == hpstream.predict_one({0: 2.0, 1: 2.0})


def test_clone_starts_fresh():
    hpstream = build_hpstream(n_samples_init=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(30):
            hpstream.learn_one({0: float(i), 1: float(i % 3)})

    clone = hpstream.clone()

    assert clone.n_clusters == 0
    assert clone.clusters == {}
    assert clone.n_spawned_clusters == 0
    assert clone._get_params() == hpstream._get_params()


def test_learning_the_original_leaves_the_clone_alone():
    hpstream = build_hpstream(n_samples_init=5)
    clone = hpstream.clone()
    snapshot = pickle.dumps(clone)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(30):
            hpstream.learn_one({0: float(i), 1: float(i % 3)})

    assert pickle.dumps(clone) == snapshot


def test_the_repr_of_a_cluster_says_what_it_holds():
    hpstream = build_hpstream()
    cluster = seed_cluster(hpstream, 0, [{0: 1.0}, {0: 3.0}], dimensions=[0])

    assert repr(cluster) == "HPStreamFadingCluster(weight=2.000, dimensions=[0])"


def test_a_cluster_without_weight_has_a_center_of_zero():
    cluster = HPStreamFadingCluster({0: 0.0}, {0: 0.0}, 0.0, 0, 0)

    assert cluster.center == {0: 0.0}
    assert cluster.radius(0) == 0.0


def test_learning_on_a_stream_of_arrays():
    hpstream = build_hpstream(n_samples_init=6, max_clusters=2, n_projected_dimensions=2)
    X = [[1, 2], [1, 3], [8, 9], [8, 8], [1, 2], [8, 9]]

    for x, _ in stream.iter_array(X):
        hpstream.learn_one(x)

    assert hpstream.n_clusters == 2
    assert hpstream.predict_one({0: 1, 1: 2}) != hpstream.predict_one({0: 8, 1: 9})


def test_memory_does_not_grow_with_the_stream():
    hpstream = build_hpstream(max_clusters=5, n_samples_init=100)
    rng = random.Random(0)
    points = [
        {0: rng.uniform(0, 10), 1: rng.uniform(0, 10), 2: rng.uniform(0, 10)} for _ in range(2000)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for point in points[:1000]:
            hpstream.learn_one(point)
        warmup = hpstream._raw_memory_usage
        for point in points[1000:]:
            hpstream.learn_one(point)

    assert hpstream._raw_memory_usage <= warmup + 1024
