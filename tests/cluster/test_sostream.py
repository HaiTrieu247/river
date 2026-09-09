from __future__ import annotations

import math
import pickle

import pytest
from sklearn.datasets import make_blobs

from river import metrics, stream, utils
from river.cluster import SOStream
from river.cluster.sostream import SOStreamMicroCluster

BLOB_CENTERS = [(-10, -10), (-5, -5), (0, 0), (5, 5), (10, 10)]
BLOB_STD = 0.6
BLOB_SAMPLES = 15_000
BLOB_SEED = 42


def build_sostream(**kwargs):
    params = {
        "alpha": 0.1,
        "min_pts": 2,
        "merge_threshold": 0.0,
        "fading_factor": 0.0,
        "fade_threshold": 0.0,
        "fading_interval": 0,
    }
    params.update(kwargs)
    return SOStream(**params)


def feed(model, values):
    for value in values:
        model.learn_one({0: value})


def build_micro_cluster(center, weight, radius):
    micro_cluster = SOStreamMicroCluster(values={0: center}, weight=weight, last_update=0)
    micro_cluster.radius = radius
    return micro_cluster


def assert_micro_cluster_properties(micro_cluster, center, weight=None, radius=None):
    assert micro_cluster.center == pytest.approx(center)
    if weight is not None:
        assert micro_cluster.weight == pytest.approx(weight)
    if radius is not None:
        assert micro_cluster.radius == pytest.approx(radius)


def test_first_min_pts_points_each_open_a_cluster():
    model = build_sostream(min_pts=3)

    feed(model, [0.0, 0.0, 0.0])

    assert len(model.micro_clusters) == 3
    for micro_cluster in model.micro_clusters.values():
        assert_micro_cluster_properties(micro_cluster, center={0: 0.0}, weight=1.0, radius=0.0)


def test_radius_is_the_min_pts_th_smallest_distinct_distance():
    model = build_sostream(min_pts=3, alpha=0.0)

    feed(model, [0.0, 1.0, 4.0, 0.0])

    assert model.micro_clusters[0].radius == pytest.approx(4.0)


def test_duplicate_distances_count_once_towards_the_radius():
    model = build_sostream(min_pts=3, alpha=0.0)

    feed(model, [0.0, 1.0, 1.0, 4.0, 0.0])

    assert model.micro_clusters[0].radius == pytest.approx(4.0)


def test_winner_absorbs_point_and_pulls_its_neighbor():
    model = build_sostream(min_pts=2, alpha=0.1)

    feed(model, [0.0, 10.0, 0.5])

    winner_radius = 10.0
    winner_center = 0.0 + (1.0 / 2.0) * (0.5 - 0.0)
    influence = math.exp(-abs(10.0 - winner_center) / (2.0 * winner_radius**2))
    neighbor_center = 10.0 + 0.1 * influence * (winner_center - 10.0)

    assert len(model.micro_clusters) == 2
    assert_micro_cluster_properties(
        model.micro_clusters[0], center={0: winner_center}, weight=2.0, radius=winner_radius
    )
    assert_micro_cluster_properties(
        model.micro_clusters[1], center={0: neighbor_center}, weight=1.0, radius=0.0
    )


def test_neighbors_are_not_pulled_when_alpha_is_zero():
    model = build_sostream(min_pts=2, alpha=0.0)

    feed(model, [0.0, 10.0, 0.5])

    assert_micro_cluster_properties(model.micro_clusters[1], center={0: 10.0}, weight=1.0)


def test_point_outside_the_winner_radius_opens_a_cluster():
    model = build_sostream(min_pts=2)

    feed(model, [0.0, 10.0, 100.0])

    assert len(model.micro_clusters) == 3
    assert model.micro_clusters[1].radius == pytest.approx(10.0)
    assert_micro_cluster_properties(
        model.micro_clusters[2], center={0: 100.0}, weight=1.0, radius=0.0
    )


def test_point_exactly_on_the_winner_radius_is_absorbed():
    model = build_sostream(min_pts=2, alpha=0.0)

    feed(model, [0.0, 10.0, -10.0])

    assert len(model.micro_clusters) == 2
    assert_micro_cluster_properties(
        model.micro_clusters[0], center={0: -5.0}, weight=2.0, radius=10.0
    )


def test_feature_order_is_pinned_independently_of_input_order():
    forward = build_sostream()
    backward = build_sostream()

    forward.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    backward.learn_one({"c": 3.0, "b": 2.0, "a": 1.0})

    assert forward._keys == backward._keys == ["a", "b", "c"]


def test_predict_one_does_not_pin_new_features():
    model = build_sostream()

    model.learn_one({"a": 1.0})
    model.predict_one({"a": 1.0, "z": 5.0})

    assert model._keys == ["a"]


def test_touching_clusters_do_not_overlap():
    model = build_sostream(min_pts=2, merge_threshold=1e9)

    feed(model, [0.0, 10.0, 100.0])

    assert len(model.micro_clusters) == 3


def test_overlapping_clusters_within_merge_threshold_are_merged():
    model = build_sostream(min_pts=2, alpha=0.1, merge_threshold=1.0)

    feed(model, [0.0, 0.1, 0.0])

    winner_radius = 0.1
    influence = math.exp(-0.1 / (2.0 * winner_radius**2))
    neighbor_center = 0.1 + 0.1 * influence * (0.0 - 0.1)
    merged_center = (2.0 * 0.0 + 1.0 * neighbor_center) / 3.0
    merged_radius = max(
        abs(merged_center - 0.0) + winner_radius,
        abs(merged_center - neighbor_center) + 0.0,
    )

    assert len(model.micro_clusters) == 1
    assert_micro_cluster_properties(
        model.micro_clusters[0], center={0: merged_center}, weight=3.0, radius=merged_radius
    )


def test_overlapping_clusters_beyond_merge_threshold_are_not_merged():
    model = build_sostream(min_pts=2, alpha=0.1, merge_threshold=0.05)

    feed(model, [0.0, 0.1, 0.0])

    assert len(model.micro_clusters) == 2


def test_tie_between_equidistant_clusters_is_broken_by_lowest_index():
    model = build_sostream(min_pts=2, alpha=0.0)

    feed(model, [0.0, 2.0, 1.0])

    assert model.micro_clusters[0].weight == pytest.approx(2.0)
    assert model.micro_clusters[1].weight == pytest.approx(1.0)
    assert model.predict_one({0: 1.0}) == 0


def test_fading_removes_clusters_below_the_fade_threshold():
    model = build_sostream(min_pts=2, fading_factor=1.0, fade_threshold=0.4)

    feed(model, [0.0, 100.0])
    model._time_stamp = 2
    model._fade_all()

    assert list(model.micro_clusters) == [1]
    assert model.micro_clusters[1].weight == pytest.approx(0.5)


def test_stream_speed_slows_the_clock():
    model = build_sostream(min_pts=2, stream_speed=10)

    feed(model, [0.0] * 25)

    assert model._n_samples_seen == 25
    assert model._time_stamp == 2


def test_stream_speed_of_one_leaves_the_clock_unchanged():
    model = build_sostream(min_pts=2)

    feed(model, [0.0] * 25)

    assert model._time_stamp == 25


def test_fading_interval_is_counted_in_samples_not_time_units():
    slow = build_sostream(min_pts=2, fading_interval=4, stream_speed=100, fade_threshold=1e9)
    fast = build_sostream(min_pts=2, fading_interval=4, stream_speed=1, fade_threshold=1e9)

    feed(slow, [float(i) for i in range(12)])
    feed(fast, [float(i) for i in range(12)])

    assert slow._time_stamp == 0
    assert fast._time_stamp == 12
    assert slow.micro_clusters == {}
    assert fast.micro_clusters == {}


def test_stream_speed_rescales_the_fading_formula():
    fading_factor = 0.25
    stream_speed = 100
    model = build_sostream(min_pts=2, fading_factor=fading_factor, stream_speed=stream_speed)

    feed(model, [0.0])
    micro_cluster = model.micro_clusters[0]
    micro_cluster.weight = 1.0
    micro_cluster.last_update = 0

    delta_samples = 400
    model._time_stamp = delta_samples // stream_speed
    model._fade_all()

    expected = 2.0 ** (-fading_factor * delta_samples / stream_speed)
    assert micro_cluster.weight == pytest.approx(expected)


def test_weight_decays_by_the_fading_formula():
    fading_factor = 0.25
    model = build_sostream(min_pts=2, fading_factor=fading_factor)
    feed(model, [0.0])
    micro_cluster = model.micro_clusters[0]

    for delta_t in (1, 5, 20):
        micro_cluster.weight = 1.0
        micro_cluster.last_update = 0
        model._time_stamp = delta_t
        model._fade_all()
        assert micro_cluster.weight == pytest.approx(2.0 ** (-fading_factor * delta_t))


def test_merged_center_is_the_weighted_mean():
    left = build_micro_cluster(center=0.0, weight=3.0, radius=1.0)
    right = build_micro_cluster(center=4.0, weight=1.0, radius=0.5)

    left.merge(right)

    assert left.weight == pytest.approx(4.0)
    assert left.center == pytest.approx({0: 1.0})


def test_merged_radius_covers_both_clusters():
    left = build_micro_cluster(center=0.0, weight=3.0, radius=1.0)
    right = build_micro_cluster(center=4.0, weight=1.0, radius=0.5)

    left.merge(right)

    assert left.radius == pytest.approx(max(1.0 + 1.0, 3.0 + 0.5))
    assert left.radius >= abs(left.center[0] - 0.0) + 1.0
    assert left.radius >= abs(left.center[0] - 4.0) + 0.5


def test_cluster_count_never_exceeds_the_number_of_points_seen():
    model = build_sostream(min_pts=2, merge_threshold=0.5)

    X, _ = make_blobs(
        n_samples=500, centers=BLOB_CENTERS, cluster_std=BLOB_STD, random_state=BLOB_SEED
    )
    for seen, (x, _) in enumerate(stream.iter_array(X), start=1):
        model.learn_one(x)
        assert len(model.micro_clusters) <= seen


def test_no_state_aliasing_with_input():
    model = build_sostream()
    x = {"a": 1.0, "b": 2.0}

    model.learn_one(x)
    before = pickle.dumps(model)
    x["a"] = 999.0

    assert pickle.dumps(model) == before


def test_emerging_features():
    model = build_sostream()

    model.learn_one({"a": 1.0})
    model.learn_one({"a": 1.0, "b": 2.0})
    model.learn_one({"b": 2.0})

    assert isinstance(model.predict_one({"a": 1.0, "b": 2.0}), int)
    for micro_cluster in model.micro_clusters.values():
        assert len(micro_cluster.center) == 2


def test_predict_one_before_any_learn_one():
    model = build_sostream()

    assert model.predict_one({"a": 1.0}) == 0


def test_shuffled_feature_order_has_no_impact():
    forward = build_sostream(merge_threshold=0.5)
    backward = build_sostream(merge_threshold=0.5)

    X, _ = make_blobs(
        n_samples=500, centers=BLOB_CENTERS, cluster_std=BLOB_STD, random_state=BLOB_SEED
    )
    for x, _ in stream.iter_array(X):
        forward.learn_one(x)
        backward.learn_one({k: x[k] for k in reversed(list(x))})

    assert forward.centers.keys() == backward.centers.keys()
    for i, center in forward.centers.items():
        assert center == pytest.approx(backward.centers[i])


def build_blobs():
    return make_blobs(
        n_samples=BLOB_SAMPLES,
        centers=BLOB_CENTERS,
        cluster_std=[BLOB_STD] * len(BLOB_CENTERS),
        n_features=2,
        random_state=BLOB_SEED,
    )


def run_blobs(model):
    X, y = build_blobs()
    v_beta = metrics.VBeta(beta=1.0)
    for x, y_true in stream.iter_array(X, y):
        model.learn_one(x)
        v_beta.update(y_true, model.predict_one(x))
    return v_beta.get()


def heaviest(model, count):
    return sorted(model.micro_clusters, key=lambda i: (-model.micro_clusters[i].weight, i))[:count]


def distance_to_nearest_blob_center(model, i):
    expected_centers = [{0: a, 1: b} for a, b in BLOB_CENTERS]
    return min(
        utils.math.minkowski_distance(model.centers[i], expected, 2)
        for expected in expected_centers
    )


def test_default_parameters_match_the_paper():
    model = SOStream()

    assert model.alpha == 0.1
    assert model.min_pts == 2
    assert model.fading_factor == 0.1
    assert model.merge_threshold == 0.1
    assert model.fade_threshold == 0.1


def test_sostream_synthetic_sklearn_with_paper_defaults():
    model = SOStream()

    v_beta = run_blobs(model)

    assert model.n_clusters == 27
    assert round(v_beta, 4) == 0.3813


def test_sostream_synthetic_sklearn_with_a_merge_threshold_matched_to_the_blobs():
    model = SOStream(merge_threshold=2.0)

    v_beta = run_blobs(model)

    assert model.n_clusters == len(BLOB_CENTERS)
    assert round(v_beta, 4) == 0.518

    for i in heaviest(model, len(BLOB_CENTERS)):
        assert distance_to_nearest_blob_center(model, i) < 0.8
