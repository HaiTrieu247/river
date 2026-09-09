from __future__ import annotations

import math
import pickle

import pytest
from sklearn.datasets import make_blobs

from river import metrics, stream, utils
from river.cluster import RepStream
from river.cluster.repstream import RepStreamVertex

BLOB_CENTERS = [(-10, -10), (-5, -5), (0, 0), (5, 5), (10, 10)]
BLOB_STD = 0.6
BLOB_SAMPLES = 15_000
BLOB_SEED = 42


def build_repstream(**kwargs):
    params = {
        "k": 3,
        "alpha": 1.5,
        "decay_rate": 0.99,
        "max_vertices": 1000,
        "repository_fraction": 0.5,
    }
    params.update(kwargs)
    return RepStream(**params)


def feed(model, values):
    for value in values:
        model.learn_one({0: float(value)})


def build_vertices(model, count):
    vertices = {}
    for i in range(count):
        vertex = RepStreamVertex(values=[float(i)], index=i, created_at=i)
        model._vertices[i] = vertex
        vertices[i] = vertex
    return vertices


def assert_vertex_properties(vertex, out=None, in_edges=None, density=None, cluster=None):
    if out is not None:
        assert vertex.out == pytest.approx(out)
    if in_edges is not None:
        assert sorted(vertex.in_edges) == sorted(in_edges)
    if density is not None:
        assert vertex.density == pytest.approx(density)
    if cluster is not None:
        assert vertex.cluster == cluster


def assert_graph_invariants(model):
    vertices = model.vertices
    for i, vertex in vertices.items():
        assert vertex.index == i
        assert len(vertex.out) <= model.k
        assert len(vertex.rep_out) <= model.k
        for j in vertex.out:
            assert i in vertices[j].in_edges
        for j in vertex.in_edges:
            assert i in vertices[j].out
        for j in vertex.rep_out:
            assert i in vertices[j].rep_in
        for j in vertex.rep_in:
            assert i in vertices[j].rep_out
        for j in vertex.dr:
            assert i in vertices[j].dr
            assert j in vertex.rep_out and i in vertices[j].rep_out
            assert vertex.is_representative and vertices[j].is_representative
        assert vertex.is_representative == (i in model._representatives)


def assert_cluster_partition(model):
    covered: set[int] = set()
    for members in model.clusters.values():
        assert members
        assert not members & covered
        covered |= members
        for i in members:
            assert model.vertices[i].cluster in model.clusters
    assert covered == set(model.vertices)


def assert_every_vertex_has_a_representative(model):
    for vertex in model.vertices.values():
        if not vertex.is_representative:
            assert model._nearest_connected_representative(vertex) is not None


def test_first_point_becomes_a_representative_of_its_own_cluster():
    model = build_repstream()

    feed(model, [0.0])

    assert sorted(model._representatives) == [0]
    assert model.n_clusters == 1
    assert_vertex_properties(model.vertices[0], out={}, in_edges=[], density=0.0, cluster=0)
    assert not model.vertices[0].is_exemplar


def test_reciprocally_connected_point_joins_the_cluster_of_its_representative():
    model = build_repstream()

    feed(model, [0.0, 10.0])

    assert sorted(model._representatives) == [0]
    assert model.n_clusters == 1
    assert_vertex_properties(model.vertices[1], out={0: 10.0}, in_edges=[0], cluster=0)
    assert model.vertices[0].density == pytest.approx(10.0)
    assert model.vertices[0].count == 1


def test_neighbour_links_back_while_below_k_neighbours():
    model = build_repstream(k=3)

    feed(model, [0.0, 100.0, 200.0])

    assert model.vertices[0].out == pytest.approx({1: 100.0, 2: 200.0})
    assert sorted(model.vertices[0].in_edges) == [1, 2]


def test_point_beyond_a_full_neighbourhood_is_not_linked_back():
    model = build_repstream(k=1)

    feed(model, [0.0, 1.0, 100.0])

    assert model.vertices[1].out == pytest.approx({0: 1.0})
    assert 2 not in model.vertices[1].out


def test_point_without_a_reciprocal_representative_opens_a_cluster():
    model = build_repstream(k=1)

    feed(model, [0.0, 1.0, 100.0])

    assert sorted(model._representatives) == [0, 2]
    assert model.n_clusters == 2
    assert model.vertices[2].cluster != model.vertices[0].cluster


def test_closer_point_evicts_the_furthest_neighbour():
    model = build_repstream(k=1)

    feed(model, [0.0, 10.0, 0.5])

    assert model.vertices[0].out == pytest.approx({2: 0.5})
    assert model.vertices[0].density == pytest.approx(0.5)


def test_evicted_neighbour_is_promoted_when_it_loses_its_only_representative():
    """Definition 7 must hold at any time; Algorithm 1 lines 32-34 visit the neighbours of
    the vertex that lost an edge but not the dropped vertex itself, which is the one that
    ends up without a reciprocally connected representative."""
    model = build_repstream(k=1)

    feed(model, [0.0, 10.0, 0.5])

    assert sorted(model._representatives) == [0, 1]
    assert model.vertices[1].is_representative
    assert model.vertices[1].cluster == model.vertices[0].cluster
    assert_every_vertex_has_a_representative(model)


def test_tie_between_equidistant_vertices_is_broken_by_lowest_index():
    model = build_repstream(k=1)

    feed(model, [0.0, 2.0, 1.0])

    assert model.vertices[2].out == pytest.approx({0: 1.0})


def test_furthest_neighbour_tie_evicts_the_highest_index():
    model = build_repstream(k=1)

    feed(model, [0.0, 1.0, -1.0])

    assert model.vertices[0].out == pytest.approx({1: 1.0})


def test_new_representative_starts_as_a_predictor():
    model = build_repstream(k=5)

    feed(model, [0, 1, 2, 3, 4, 5, 50, 51, 52, 53, 54, 55])

    assert sorted(model._representatives) == [0, 6]
    for index in model._representatives:
        assert len(model.vertices[index].rep_in) == 1
        assert not model.vertices[index].is_exemplar
    assert sorted(model.predictors) == [0, 6]
    assert model.exemplars == {}


def test_predictor_is_upgraded_once_half_of_k_edges_point_at_it():
    model = build_repstream(k=5)

    feed(model, [0, 1, 2, 3, 4, 5, 50, 51, 52, 53, 54, 55, 100, 101, 102, 103, 104, 105])

    assert sorted(model._representatives) == [0, 6, 12]
    for index in model._representatives:
        assert 2 * len(model.vertices[index].rep_in) < model.k
        assert not model.vertices[index].is_exemplar

    feed(model, [150, 151, 152, 153, 154, 155])

    assert sorted(model._representatives) == [0, 6, 12, 18]
    for index in model._representatives:
        assert 2 * len(model.vertices[index].rep_in) >= model.k
        assert model.vertices[index].is_exemplar
    assert model.predictors == {}


def test_density_is_the_mean_distance_to_the_nearest_neighbours():
    model = build_repstream(k=2)

    feed(model, [0.0, 1.0, 2.0])

    assert model.vertices[0].out == pytest.approx({1: 1.0, 2: 2.0})
    assert model.vertices[0].density == pytest.approx(1.5)


def test_density_related_link_merges_two_clusters():
    model = build_repstream(k=2, alpha=2.0)

    feed(model, [0.0, 1.0, 2.0, 3.0])

    assert model.vertices[0].density == pytest.approx(1.5)
    assert model.vertices[3].density == pytest.approx(1.5)
    assert model.vertices[0].rep_out[3] == pytest.approx(3.0)
    assert model.vertices[0].rep_out[3] <= model.vertices[0].density * model.alpha
    assert model.vertices[0].rep_out[3] <= model.vertices[3].density * model.alpha
    assert sorted(model.vertices[0].dr) == [3]
    assert sorted(model.vertices[3].dr) == [0]
    assert model.n_clusters == 1
    assert sorted(model.clusters[0]) == [0, 1, 2, 3]


def test_reciprocal_representatives_below_the_density_scaler_do_not_merge():
    model = build_repstream(k=2, alpha=1.9)

    feed(model, [0.0, 1.0, 2.0, 3.0])

    assert model.vertices[0].rep_out[3] == pytest.approx(3.0)
    assert model.vertices[0].rep_out[3] > model.vertices[0].density * model.alpha
    assert model.vertices[0].dr == set()
    assert model.n_clusters == 2


def test_shrinking_density_drops_the_link_and_splits_the_cluster():
    model = build_repstream(k=2, alpha=2.0)

    feed(model, [0.0, 1.0, 2.0, 3.0])
    assert model.n_clusters == 1

    feed(model, [4.0])

    assert model.vertices[3].density == pytest.approx(1.0)
    assert model.vertices[0].rep_out[3] == pytest.approx(3.0)
    assert model.vertices[0].rep_out[3] > model.vertices[3].density * model.alpha
    assert model.vertices[0].dr == set()
    assert model.n_clusters == 2
    assert sorted(model.clusters[model.vertices[0].cluster]) == [0, 1]
    assert sorted(model.clusters[model.vertices[3].cluster]) == [2, 3, 4]


def test_split_keeps_the_largest_component_in_the_original_cluster():
    model = build_repstream(k=2, alpha=2.0)

    feed(model, [0.0, 1.0, 2.0, 3.0])
    original = model.vertices[0].cluster

    feed(model, [4.0])

    assert model.vertices[0].cluster == original
    assert model.vertices[3].cluster != original


def test_identical_points_form_a_singularity():
    model = build_repstream(k=2)

    feed(model, [0.0, 0.0, 0.0])

    assert sorted(model._singularities) == [0]
    assert model.vertices[0].density == 0.0
    assert len(model.vertices[0].out) == model.k
    assert math.fsum(model.vertices[0].out.values()) == 0.0


def test_point_identical_to_a_singularity_is_absorbed():
    model = build_repstream(k=2)

    feed(model, [0.0, 0.0, 0.0])
    before = len(model.vertices)
    count = model.vertices[0].count

    feed(model, [0.0, 0.0, 0.0])

    assert len(model.vertices) == before
    assert model.vertices[0].count == count + 3


def test_singularity_keeps_a_zero_density_and_forms_no_density_related_link():
    model = build_repstream(k=2, alpha=1e9)

    feed(model, [0.0, 0.0, 0.0, 5.0, 6.0, 7.0])

    assert 0 in model._singularities
    assert model.vertices[0].density == 0.0
    assert model.vertices[0].dr == set()
    assert model.n_clusters > 1


def test_repository_admits_new_representatives_while_there_is_room():
    model = build_repstream(k=1, max_vertices=6, repository_fraction=0.5)
    vertices = build_vertices(model, 3)

    assert model._repository_capacity == 3
    for i in range(3):
        assert model._update_repository(vertices[i], 1) is None
    assert sorted(model._repository) == [0, 1, 2]


def test_full_repository_evicts_the_least_useful_member():
    model = build_repstream(k=1, max_vertices=6, repository_fraction=0.5)
    vertices = build_vertices(model, 4)
    for i in range(3):
        model._update_repository(vertices[i], 1)

    deleted = model._update_repository(vertices[3], 1)

    assert deleted == 0
    assert sorted(model._repository) == [1, 2, 3]


def test_reinforcement_count_keeps_an_old_representative_in_the_repository():
    model = build_repstream(k=1, max_vertices=6, repository_fraction=0.5)
    vertices = build_vertices(model, 4)
    model._update_repository(vertices[0], 50)
    model._update_repository(vertices[1], 1)
    model._update_repository(vertices[2], 1)

    deleted = model._update_repository(vertices[3], 1)

    assert deleted == 1
    assert sorted(model._repository) == [0, 2, 3]


def test_usefulness_follows_the_paper_decay_function():
    model = build_repstream(decay_rate=0.5)
    vertices = build_vertices(model, 1)
    model._time_stamp = 10

    expected = math.log(0.5) * (10 - 0 + 1) + math.log(4 + 1)

    assert model._usefulness(vertices[0], 4) == pytest.approx(expected)


def test_repository_key_orders_the_same_way_as_usefulness():
    model = build_repstream()
    vertices = build_vertices(model, 6)
    model._time_stamp = 20
    for i, count in enumerate([0, 3, 1, 9, 2, 5]):
        vertices[i].count = count

    for left in vertices.values():
        for right in vertices.values():
            key_order = model._repository_key(left) < model._repository_key(right)
            usefulness_order = model._usefulness(left, left.count) < model._usefulness(
                right, right.count
            )
            assert key_order == usefulness_order


def test_vertex_count_never_exceeds_the_budget():
    model = build_repstream(k=3, max_vertices=25)

    for seen, value in enumerate(range(300), start=1):
        model.learn_one({0: float(value % 37)})
        assert len(model.vertices) <= 25
        assert len(model.vertices) <= seen


def test_repository_never_exceeds_its_capacity():
    model = build_repstream(k=3, max_vertices=20, repository_fraction=0.5)

    for value in range(300):
        model.learn_one({0: float(value % 41)})
        assert len(model._repository) <= model._repository_capacity
        assert set(model._repository) <= set(model.vertices)


def test_repository_members_survive_the_deletion_queue():
    model = build_repstream(k=3, max_vertices=20, repository_fraction=0.5)

    for value in range(300):
        model.learn_one({0: float(value % 41)})

    assert model._repository
    for index in model._repository:
        assert index in model.vertices
        assert model.vertices[index].is_representative


def test_graph_stays_consistent_along_the_stream():
    model = build_repstream(k=3, alpha=1.5, max_vertices=40)

    X, _ = make_blobs(
        n_samples=600, centers=BLOB_CENTERS, cluster_std=BLOB_STD, random_state=BLOB_SEED
    )
    for x, _ in stream.iter_array(X):
        model.learn_one(x)
        assert_graph_invariants(model)
        assert_cluster_partition(model)
        assert_every_vertex_has_a_representative(model)


def test_clusters_are_connected_components_of_density_related_links():
    model = build_repstream(k=5, alpha=3.0, max_vertices=60)

    X, _ = make_blobs(
        n_samples=600, centers=BLOB_CENTERS, cluster_std=BLOB_STD, random_state=BLOB_SEED
    )
    for x, _ in stream.iter_array(X):
        model.learn_one(x)

    for members in model.clusters.values():
        representatives = [i for i in members if model.vertices[i].is_representative]
        if len(representatives) <= 1:
            continue
        seen = {representatives[0]}
        stack = [representatives[0]]
        while stack:
            i = stack.pop()
            for j in model.vertices[i].dr:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        assert seen == set(representatives)


def test_no_state_aliasing_with_input():
    model = build_repstream()
    x = {"a": 1.0, "b": 2.0}

    model.learn_one(x)
    before = pickle.dumps(model)
    x["a"] = 999.0

    assert pickle.dumps(model) == before


def test_emerging_features():
    model = build_repstream()

    model.learn_one({"a": 1.0})
    model.learn_one({"a": 1.0, "b": 2.0})
    model.learn_one({"b": 2.0})

    assert isinstance(model.predict_one({"a": 1.0, "b": 2.0}), int)
    for vertex in model.vertices.values():
        assert len(vertex.values) == 2


def test_predict_one_before_any_learn_one():
    model = build_repstream()

    assert model.predict_one({"a": 1.0}) == 0


def test_predict_one_does_not_pin_new_features():
    model = build_repstream()

    model.learn_one({"a": 1.0})
    model.predict_one({"a": 1.0, "z": 5.0})

    assert model._keys == ["a"]


def test_feature_order_is_pinned_independently_of_input_order():
    forward = build_repstream()
    backward = build_repstream()

    forward.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    backward.learn_one({"c": 3.0, "b": 2.0, "a": 1.0})

    assert forward._keys == backward._keys == ["a", "b", "c"]


def test_shuffled_feature_order_has_no_impact():
    forward = build_repstream(k=3, max_vertices=40)
    backward = build_repstream(k=3, max_vertices=40)

    X, _ = make_blobs(
        n_samples=400, centers=BLOB_CENTERS, cluster_std=BLOB_STD, random_state=BLOB_SEED
    )
    for x, _ in stream.iter_array(X):
        forward.learn_one(x)
        backward.learn_one({k: x[k] for k in reversed(list(x))})

    assert forward.clusters == backward.clusters
    assert forward.centers.keys() == backward.centers.keys()
    for cluster, center in forward.centers.items():
        assert center == pytest.approx(backward.centers[cluster])


def test_predict_one_returns_the_cluster_of_the_nearest_representative():
    model = build_repstream(k=1)

    feed(model, [0.0, 1.0, 100.0])

    assert model.predict_one({0: 0.2}) == model.vertices[0].cluster
    assert model.predict_one({0: 99.0}) == model.vertices[2].cluster


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


def distance_to_nearest_blob_center(center):
    return min(utils.math.minkowski_distance(center, {0: a, 1: b}, 2) for a, b in BLOB_CENTERS)


def test_default_parameters_match_the_paper():
    model = RepStream()

    assert model.k == 5
    assert model.alpha == 1.5
    assert model.decay_rate == 0.99
    assert model.repository_fraction == 0.5


def test_repstream_synthetic_sklearn_with_paper_defaults():
    model = RepStream()

    v_beta = run_blobs(model)

    assert model.n_clusters == 20
    assert round(v_beta, 4) == 0.3635


def test_repstream_synthetic_sklearn_with_a_density_scaler_matched_to_the_blobs():
    model = RepStream(k=7, alpha=3.0)

    v_beta = run_blobs(model)

    assert model.n_clusters == len(BLOB_CENTERS)
    assert round(v_beta, 4) == 0.9917

    for center in model.centers.values():
        assert distance_to_nearest_blob_center(center) < 0.3
