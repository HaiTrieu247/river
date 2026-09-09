from __future__ import annotations

import math
import pickle
import random

import pytest
from sklearn.datasets import make_blobs

from river import metrics, stream, utils
from river.cluster import BIRCH, ImprovedBIRCH
from river.cluster.improved_birch import (
    _MEASURES,
    ImprovedBIRCHClusteringFeature,
    _summary,
)

BLOB_CENTERS = [(-10, -10), (-5, -5), (0, 0), (5, 5), (10, 10)]
BLOB_STD = 0.6
BLOB_SAMPLES = 15_000
BLOB_SEED = 42


def build_improved_birch(**kwargs):
    params = {
        "initial_threshold": 1.0,
        "modifying_factor": 2.0,
        "branching_factor": 3,
        "leaf_capacity": 3,
        "max_entries": 10**6,
    }
    params.update(kwargs)
    return ImprovedBIRCH(**params)


def feed(model, values):
    for value in values:
        model.learn_one({0: float(value)})
    return model


def build_feature(points, threshold=0.0):
    feature = ImprovedBIRCHClusteringFeature(threshold=threshold)
    for point in points:
        feature += ImprovedBIRCHClusteringFeature.from_point(point)
    return feature


def entry_centers(model):
    return [feature.center[0] for feature in model.micro_clusters.values()]


def entry_thresholds(model):
    return list(model.thresholds.values())


def assert_feature_properties(feature, center, n=None, threshold=None):
    assert feature.center == pytest.approx(center)
    if n is not None:
        assert feature.n == pytest.approx(n)
    if threshold is not None:
        assert feature.threshold == pytest.approx(threshold)


def assert_tree_is_valid(model):
    depths = set()
    stack = [(model.root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.leaf:
            assert len(node.features) <= model.leaf_capacity
            depths.add(depth)
            continue
        assert len(node.features) <= model.branching_factor
        assert len(node.features) == len(node.children)
        for feature, child in zip(node.features, node.children):
            summary = _summary(child.features)
            assert feature.n == pytest.approx(summary.n)
            assert feature.linear_sum == pytest.approx(summary.linear_sum)
            assert feature.squared_sum == pytest.approx(summary.squared_sum)
            stack.append((child, depth + 1))
    assert len(depths) == 1


def test_first_point_opens_a_leaf_entry_at_the_initial_threshold():
    model = build_improved_birch(initial_threshold=1.0)

    model.learn_one({0: 0.0})

    assert len(model.micro_clusters) == 1
    assert model.height == 1
    assert_feature_properties(model.micro_clusters[0], center=[0.0], n=1.0, threshold=1.0)


def test_entry_absorbs_within_its_own_threshold_and_keeps_it():
    model = feed(build_improved_birch(initial_threshold=1.0), [0.0, 1.0])

    assert len(model.micro_clusters) == 1
    assert_feature_properties(model.micro_clusters[0], center=[0.5], n=2.0, threshold=1.0)
    assert model.n_threshold_growths == 0
    assert model.n_neighbor_merges == 0


def test_entry_absorbs_with_the_grown_threshold_and_keeps_the_new_one():
    model = feed(build_improved_birch(initial_threshold=1.0), [0.0, 1.0, 2.5])

    assert len(model.micro_clusters) == 1
    assert_feature_properties(model.micro_clusters[0], center=[7.0 / 6.0], n=3.0, threshold=2.0)
    assert model.n_threshold_growths == 1
    assert model.n_neighbor_merges == 0


def test_grown_threshold_absorbs_the_nearest_neighbour_entry():
    model = feed(build_improved_birch(initial_threshold=1.0, leaf_capacity=10), [0.0, 2.4, 1.2])

    assert len(model.micro_clusters) == 1
    assert_feature_properties(model.micro_clusters[0], center=[1.2], n=3.0, threshold=2.0)
    assert model.n_threshold_growths == 1
    assert model.n_neighbor_merges == 1


def test_neighbour_out_of_reach_of_the_grown_threshold_survives():
    model = feed(build_improved_birch(initial_threshold=0.5, leaf_capacity=10), [0.0, 2.0, 1.0])

    assert len(model.micro_clusters) == 2
    assert_feature_properties(model.micro_clusters[0], center=[0.5], n=2.0, threshold=1.0)
    assert_feature_properties(model.micro_clusters[1], center=[2.0], n=1.0, threshold=0.5)
    assert model.n_threshold_growths == 1
    assert model.n_neighbor_merges == 0


def test_point_beyond_the_grown_threshold_opens_an_entry_and_keeps_the_old_one():
    model = feed(build_improved_birch(initial_threshold=1.0), [0.0, 5.0])

    assert len(model.micro_clusters) == 2
    assert entry_centers(model) == [0.0, 5.0]
    assert entry_thresholds(model) == [1.0, 1.0]
    assert model.n_threshold_growths == 0


def test_new_entries_still_start_at_the_initial_threshold_after_a_growth():
    model = feed(build_improved_birch(initial_threshold=1.0), [0.0, 1.0, 2.5, 9.0])

    assert entry_centers(model) == pytest.approx([7.0 / 6.0, 9.0])
    assert entry_thresholds(model) == [2.0, 1.0]


def test_thresholds_diverge_across_leaf_entries():
    generator = random.Random(4)
    model = build_improved_birch(initial_threshold=0.4, branching_factor=6, leaf_capacity=6)

    for _ in range(500):
        model.learn_one({0: generator.uniform(0, 20)})

    assert len(set(entry_thresholds(model))) > 1
    assert min(entry_thresholds(model)) == pytest.approx(0.4)


def test_a_tie_between_equidistant_entries_is_broken_by_lowest_index():
    model = feed(build_improved_birch(initial_threshold=0.5, leaf_capacity=10), [0.0, 2.0, 1.0])

    assert_feature_properties(model.micro_clusters[0], center=[0.5], n=2.0)
    assert_feature_properties(model.micro_clusters[1], center=[2.0], n=1.0)


def test_the_effective_modifying_factor_decreases_as_the_threshold_grows():
    model = build_improved_birch(initial_threshold=0.5, modifying_factor=2.0)
    thresholds = [0.5]
    for _ in range(4):
        thresholds.append(model._grown(thresholds[-1]))

    factors = [b / a for a, b in zip(thresholds, thresholds[1:])]

    assert thresholds == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])
    assert factors == pytest.approx([2.0, 1.5, 4.0 / 3.0, 1.25])
    assert factors == sorted(factors, reverse=True)


def test_a_modifying_factor_of_one_freezes_every_threshold():
    model = feed(build_improved_birch(initial_threshold=1.0, modifying_factor=1.0), [0.0, 1.0, 2.5])

    assert model.threshold_step == 0.0
    assert len(model.micro_clusters) == 2
    assert entry_thresholds(model) == [1.0, 1.0]
    assert model.n_threshold_growths == 0


def test_the_neighbour_merge_can_push_an_entry_past_its_own_threshold():
    model = feed(
        build_improved_birch(initial_threshold=1.0, modifying_factor=3.0, leaf_capacity=10),
        [1.477, 5.243, 5.989, 0.933, 5.234, 2.945],
    )

    assert len(model.micro_clusters) == 1
    feature = model.micro_clusters[0]
    assert feature.n == pytest.approx(6.0)
    assert feature.threshold == pytest.approx(3.0)
    assert feature.diameter() == pytest.approx(3.040897, abs=1e-6)
    assert feature.diameter() > feature.threshold


def test_the_neighbour_merge_preserves_the_path_summaries():
    generator = random.Random(5)
    model = build_improved_birch(initial_threshold=0.3, branching_factor=4, leaf_capacity=4)

    for _ in range(800):
        model.learn_one({0: generator.uniform(0, 10), 1: generator.uniform(0, 10)})

    assert model.n_neighbor_merges > 0
    assert_tree_is_valid(model)
    assert sum(f.n for f in model.micro_clusters.values()) == pytest.approx(800)


def test_full_leaf_splits_on_the_farthest_pair():
    model = feed(
        build_improved_birch(initial_threshold=0.5, leaf_capacity=3),
        [0.0, 10.0, 20.0, 30.0],
    )

    assert model.height == 2
    assert [len(child.features) for child in model.root.children] == [2, 2]
    assert entry_centers(model) == [0.0, 10.0, 30.0, 20.0]
    assert entry_thresholds(model) == [0.5, 0.5, 0.5, 0.5]


def test_root_split_increases_the_height():
    model = feed(
        build_improved_birch(initial_threshold=0.1, modifying_factor=1.5),
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],
    )

    assert model.height == 3
    assert len(model.micro_clusters) == 12
    assert entry_thresholds(model) == [0.1] * 12
    assert_tree_is_valid(model)


def test_path_features_summarise_their_children_on_a_random_stream():
    generator = random.Random(0)
    model = build_improved_birch(initial_threshold=0.5, branching_factor=4, leaf_capacity=4)

    for _ in range(1_000):
        model.learn_one({0: generator.uniform(0, 10), 1: generator.uniform(0, 10)})

    assert_tree_is_valid(model)
    assert sum(f.n for f in model.micro_clusters.values()) == pytest.approx(1_000)


def test_rebuild_grows_every_threshold_and_shrinks_the_tree():
    model = build_improved_birch(
        initial_threshold=0.5,
        leaf_capacity=5,
        branching_factor=5,
        max_entries=10,
        outlier_buffer_size=0,
    )

    feed(model, [value * 100.0 for value in range(15)])

    assert model.n_rebuilds == 1
    assert len(model.micro_clusters) <= 10
    assert max(entry_thresholds(model)) == pytest.approx(100.0)


def test_a_point_after_a_rebuild_still_opens_an_entry_at_the_initial_threshold():
    model = build_improved_birch(
        initial_threshold=0.5,
        leaf_capacity=5,
        branching_factor=5,
        max_entries=12,
        outlier_buffer_size=0,
    )

    feed(model, [value * 100.0 for value in range(15)])
    assert model.n_rebuilds == 1
    assert max(entry_thresholds(model)) == pytest.approx(100.0)
    before = entry_thresholds(model).count(0.5)

    feed(model, [-5_000.0])

    assert model.n_rebuilds == 1
    assert entry_thresholds(model).count(0.5) == before + 1
    assert max(entry_thresholds(model)) == pytest.approx(100.0)


def test_leaf_entries_stay_within_the_budget():
    generator = random.Random(2)
    model = build_improved_birch(
        initial_threshold=0.5, leaf_capacity=5, branching_factor=5, max_entries=25
    )

    for _ in range(2_000):
        model.learn_one({0: generator.uniform(0, 100)})
        assert len(model.micro_clusters) <= 25


def test_a_frozen_threshold_cannot_enforce_the_budget():
    model = build_improved_birch(
        initial_threshold=0.5,
        modifying_factor=1.0,
        leaf_capacity=5,
        branching_factor=5,
        max_entries=10,
    )

    feed(model, range(11))

    assert model.threshold_step == 0.0
    assert model.n_rebuilds == 0
    assert len(model.micro_clusters) == 11


def test_duplicate_entries_do_not_stall_the_rebuild():
    model = build_improved_birch(
        initial_threshold=0.1, leaf_capacity=4, branching_factor=4, max_entries=12
    )

    for _ in range(30):
        feed(model, [0.0, 0.1, 5.0, 5.1])

    assert len(model.micro_clusters) <= 12
    assert sum(f.n for f in model.micro_clusters.values()) + sum(
        f.n for f in model.outliers
    ) == pytest.approx(120.0)


def test_sparse_entries_are_written_to_the_outlier_buffer():
    model = build_improved_birch(
        initial_threshold=0.02,
        leaf_capacity=4,
        branching_factor=4,
        max_entries=12,
        outlier_fraction=0.25,
        outlier_buffer_size=5,
    )

    for value in range(10):
        feed(model, [value] * 100)
    feed(model, [1_000.0, 2_000.0, 3_000.0])

    assert model.n_rebuilds >= 1
    assert [feature.center[0] for feature in model.outliers] == [1_000.0, 2_000.0, 3_000.0]
    assert 1_000.0 not in entry_centers(model)


def test_a_zero_buffer_disables_outlier_handling():
    model = build_improved_birch(
        initial_threshold=0.1,
        leaf_capacity=4,
        branching_factor=4,
        max_entries=12,
        outlier_buffer_size=0,
    )

    for value in range(10):
        feed(model, [value] * 100)
    feed(model, [1_000.0, 2_000.0, 3_000.0])

    assert model.outliers == []
    assert sum(f.n for f in model.micro_clusters.values()) == pytest.approx(1003.0)


def test_without_a_phase_three_target_the_clusters_are_the_leaf_entries():
    model = feed(build_improved_birch(initial_threshold=0.1), [0.0, 1.0, 10.0])

    assert model.n_clusters == 3
    assert [feature.center[0] for feature in model.clusters.values()] == entry_centers(model)


def test_recluster_merges_down_to_the_requested_number_of_clusters():
    model = build_improved_birch(
        initial_threshold=0.1, leaf_capacity=10, branching_factor=10, n_macro_clusters=2
    )

    feed(model, [0.0, 0.5, 1.0, 10.0, 10.5, 11.0])

    assert model.n_clusters == 2
    centers = sorted(feature.center[0] for feature in model.clusters.values())
    assert centers == pytest.approx([0.5, 10.5])
    assert sorted(feature.n for feature in model.clusters.values()) == [3.0, 3.0]


def test_recluster_leaves_the_tree_untouched():
    model = build_improved_birch(
        initial_threshold=0.1, leaf_capacity=10, branching_factor=10, n_macro_clusters=2
    )

    feed(model, [0.0, 0.5, 1.0, 10.0, 10.5, 11.0])
    before = entry_centers(model)
    thresholds = entry_thresholds(model)
    model._recluster()

    assert entry_centers(model) == before
    assert entry_thresholds(model) == thresholds


def test_clustering_feature_is_additive():
    left = build_feature([[0.0, 1.0], [2.0, 3.0]])
    right = build_feature([[4.0, 5.0]])
    pooled = build_feature([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    merged = left + right

    assert merged.n == pytest.approx(pooled.n)
    assert merged.linear_sum == pytest.approx(pooled.linear_sum)
    assert merged.squared_sum == pytest.approx(pooled.squared_sum)
    assert merged.center == pytest.approx(pooled.center)


def test_merging_keeps_the_absorbing_threshold():
    left = build_feature([[0.0]], threshold=2.0)
    right = build_feature([[1.0]], threshold=7.0)

    left += right

    assert left.threshold == pytest.approx(2.0)
    assert (left + right).threshold == pytest.approx(2.0)


def test_merged_center_matches_the_pooled_data():
    left_points = [[0.0], [0.2], [0.4]]
    right_points = [[0.6], [0.8]]
    left = build_feature(left_points)
    right = build_feature(right_points)
    pooled = build_feature(left_points + right_points)

    assert left.merged_radius(right) == pytest.approx(pooled.radius())
    assert left.merged_diameter(right) == pytest.approx(pooled.diameter())
    left += right
    assert left.center == pytest.approx(pooled.center)


def test_radius_and_diameter_match_their_definitions():
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]
    feature = build_feature(points)
    center = feature.center

    radius = math.sqrt(sum(math.dist(p, center) ** 2 for p in points) / len(points))
    pairs = sum(math.dist(p, q) ** 2 for p in points for q in points)
    diameter = math.sqrt(pairs / (len(points) * (len(points) - 1)))

    assert feature.radius() == pytest.approx(radius)
    assert feature.diameter() == pytest.approx(diameter)


def test_distance_measures_match_their_definitions():
    left_points = [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
    right_points = [[5.0, 5.0], [6.0, 7.0]]
    left = build_feature(left_points)
    right = build_feature(right_points)
    pooled = build_feature(left_points + right_points)

    left_center = left.center
    right_center = right.center
    cross = [math.dist(p, q) ** 2 for p in left_points for q in right_points]

    def sum_of_squares(points):
        center = [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))]
        return sum(math.dist(p, center) ** 2 for p in points)

    assert _MEASURES["d0"](left, right) == pytest.approx(math.dist(left_center, right_center))
    assert _MEASURES["d1"](left, right) == pytest.approx(
        sum(abs(a - b) for a, b in zip(left_center, right_center))
    )
    assert _MEASURES["d2"](left, right) == pytest.approx(math.sqrt(sum(cross) / len(cross)))
    assert _MEASURES["d3"](left, right) == pytest.approx(pooled.diameter())
    assert _MEASURES["d4"](left, right) == pytest.approx(
        math.sqrt(
            sum_of_squares(left_points + right_points)
            - sum_of_squares(left_points)
            - sum_of_squares(right_points)
        )
    )


def test_every_distance_measure_builds_a_valid_tree():
    generator = random.Random(3)
    points = [{0: generator.uniform(0, 10), 1: generator.uniform(0, 10)} for _ in range(400)]

    for measure in ("d0", "d1", "d2", "d3", "d4"):
        model = build_improved_birch(
            initial_threshold=0.5,
            branching_factor=4,
            leaf_capacity=4,
            distance_measure=measure,
        )
        for point in points:
            model.learn_one(point)
        assert_tree_is_valid(model)
        assert sum(f.n for f in model.micro_clusters.values()) == pytest.approx(400)


def test_the_radius_threshold_measure_builds_a_valid_tree():
    generator = random.Random(6)
    model = build_improved_birch(
        initial_threshold=0.5,
        branching_factor=4,
        leaf_capacity=4,
        threshold_measure="radius",
    )

    for _ in range(400):
        model.learn_one({0: generator.uniform(0, 10), 1: generator.uniform(0, 10)})

    assert_tree_is_valid(model)
    assert sum(f.n for f in model.micro_clusters.values()) == pytest.approx(400)


def test_no_state_aliasing_with_input():
    model = build_improved_birch()
    x = {"a": 1.0, "b": 2.0}

    model.learn_one(x)
    before = pickle.dumps(model)
    x["a"] = 999.0

    assert pickle.dumps(model) == before


def test_emerging_features():
    model = build_improved_birch()

    model.learn_one({"a": 1.0})
    model.learn_one({"a": 1.0, "b": 2.0})
    model.learn_one({"b": 2.0})

    assert isinstance(model.predict_one({"a": 1.0, "b": 2.0}), int)
    for feature in model.micro_clusters.values():
        assert len(feature.linear_sum) == 2


def test_predict_one_before_any_learn_one():
    model = build_improved_birch()

    assert model.predict_one({"a": 1.0}) == 0


def test_shuffled_feature_order_has_no_impact():
    forward = build_improved_birch(initial_threshold=1.0, branching_factor=5, leaf_capacity=5)
    backward = build_improved_birch(initial_threshold=1.0, branching_factor=5, leaf_capacity=5)

    X, _ = make_blobs(
        n_samples=500, centers=BLOB_CENTERS, cluster_std=BLOB_STD, random_state=BLOB_SEED
    )
    for x, _ in stream.iter_array(X):
        forward.learn_one(x)
        backward.learn_one({k: x[k] for k in reversed(list(x))})

    assert forward.centers.keys() == backward.centers.keys()
    for i, center in forward.centers.items():
        assert center == pytest.approx(backward.centers[i])


def test_the_tree_is_smaller_than_the_basic_birch_tree_at_the_same_threshold():
    X, y = make_blobs(
        n_samples=BLOB_SAMPLES,
        centers=BLOB_CENTERS,
        cluster_std=[BLOB_STD] * len(BLOB_CENTERS),
        n_features=2,
        random_state=BLOB_SEED,
    )

    basic = BIRCH(threshold=0.5, branching_factor=7, leaf_capacity=7, max_entries=10**9)
    improved = ImprovedBIRCH(
        initial_threshold=0.5, branching_factor=7, leaf_capacity=7, max_entries=10**9
    )
    for x, _ in stream.iter_array(X, y):
        basic.learn_one(x)
        improved.learn_one(x)

    assert len(basic.micro_clusters) == 201
    assert len(improved.micro_clusters) == 7
    assert improved.n_threshold_growths > 0
    assert improved.n_neighbor_merges > 0


def test_improved_birch_synthetic_sklearn():
    X, y = make_blobs(
        n_samples=BLOB_SAMPLES,
        centers=BLOB_CENTERS,
        cluster_std=[BLOB_STD] * len(BLOB_CENTERS),
        n_features=2,
        random_state=BLOB_SEED,
    )

    model = ImprovedBIRCH()
    for x, _ in stream.iter_array(X, y):
        model.learn_one(x)

    assert model.n_rebuilds == 0
    assert len(model.micro_clusters) == 5
    assert entry_thresholds(model) == [1.5] * 5
    assert sum(f.n for f in model.micro_clusters.values()) == pytest.approx(BLOB_SAMPLES)
    assert_tree_is_valid(model)


def test_improved_birch_synthetic_sklearn_with_phase_three():
    X, y = make_blobs(
        n_samples=BLOB_SAMPLES,
        centers=BLOB_CENTERS,
        cluster_std=[BLOB_STD] * len(BLOB_CENTERS),
        n_features=2,
        random_state=BLOB_SEED,
    )

    model = ImprovedBIRCH(n_macro_clusters=len(BLOB_CENTERS))
    for x, _ in stream.iter_array(X, y):
        model.learn_one(x)

    v_beta = metrics.VBeta(beta=1.0)
    for x, y_true in stream.iter_array(X, y):
        v_beta.update(y_true, model.predict_one(x))

    assert model.n_clusters == len(BLOB_CENTERS)
    assert round(v_beta.get(), 4) == 1.0

    expected_centers = [{0: a, 1: b} for a, b in BLOB_CENTERS]
    for center in model.centers.values():
        assert (
            min(utils.math.minkowski_distance(center, expected, 2) for expected in expected_centers)
            < 0.05
        )
