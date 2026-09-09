from __future__ import annotations

import collections
import heapq
import math
import operator

from river import base


class RepStream(base.Clusterer):
    """RepStream

    The RepStream algorithm is a sparse-graph-based stream clustering approach
    that employs representative cluster points to incrementally process incoming data.
    The graph-based description allows modelling spatio-temporal relationships
    in a data stream more accurately than is possible via summary statistics.
    Each cluster is defined by using two types of representative points:

    * Exemplar points that are used to capture the stable properties of the cluster.

    * Predictor points which are used to capture the evolving properties of the cluster.

    It also avoids re-discovery of ” previously learned patterns by maximising the
    reuse of previously useful cluster information, which is captured in a repository
    of representative points.

    The use of repository offers two major benefits:

    * Effectively handle recurrent changes in the clusters, by storing a concise
    representation of persistent and consistent cluster features.

    * Provides a concise knowledge collection that can be used to rebuild a
    cluster’s overall shape and data distribution history. Therefore, it is possible
    to archive core cluster features when a recall of historical changes is desired.

    For a new point `p`:

    * Insert `p` into the sparse graph and update the graph structure. Find the `k`
    nearest neighbors of `p` and link them to `p`. Update SG's edges.

    * Checking reciprocal connections between `p` and its neighbors. If there is any
    reciprocal connections, then `p` is a ordinary point. Otherwise, `p` is a representative
    point.

    * In case `p` is a representative point, it becomes r. Check if it is predictor
    or exemplar and update the RSG structure. Calculate the relative density of `r` RD(r).
    Perform the process of merging base on Density-Related Connection Approach.

    * Update the repository to store the new representative point `r`. It is positioned
    in the repository based on its usefulness

    Because of the Memory Constraint, the number of vertices in the sparse graph is limited.
    If the number of vertices exceeds the limit, then ordinary points will be removed by
    FIFO approach. Representative points will last longer for future patterns but can also be
    removed if they are not useful.

    Parameters
    ----------
    k
        The number of nearest neighbors to consider when constructing the sparse graph.

    alpha
        Determines the threshold for density-related connections between representative points.

    decay_rate
        This parameter is used to calculate the usefulness of representative points in the repository.

    max_vertices
        The maximum number of vertices allowed in the sparse graph, which is an alternative method to
        control the memory usage of the algorithm.

    repository_fraction
        Visualizing the ratio between ordinary points and representative points in the sparse graph.
        This allow the algorithm to control the performance of removing vertices due to to memory constraint.

    distance_measure
        There are two types of distance measures available: "euclidean" and "manhattan". For convenience,
        we perform it as a parameter to allow users to choose the distance measure that is suitable for their data.


    References
    ----------
    [^1]: Lühr S, Lazarescu M (2009) Incremental clustering of dynamic data streams using connectivity based
    representative points. DataKnowl Eng 68(1):1-27.

    Examples
    --------

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 0.5], [1, 0.625], [1, 0.75], [1, 1.125], [1, 1.5], [1, 1.75],
    ...     [4, 1.5], [4, 2.25], [4, 2.5], [4, 3], [4, 3.25], [4, 3.5]
    ... ]

    >>> repstream = cluster.RepStream(
    ...     k=5,
    ...     alpha=1.5,
    ...     decay_rate=0.99,
    ...     max_vertices=100,
    ...     repository_fraction=0.5,
    ...     distance_measure="manhattan"
    ... )

    >>> for x, _ in stream.iter_array(X):
    ...     repstream.learn_one(x)

    >>> repstream.n_clusters
    2

    >>> repstream.predict_one({0: 1, 1: 2})
    0

    >>> repstream.predict_one({0: 4, 1: 3})
    1

    >>> sorted(repstream.micro_clusters)
    [0, 6]

    >>> sorted(repstream.predictors)
    [0, 6]

    >>> repstream.exemplars
    {}

    >>> repeated = cluster.RepStream(
    ...     k=5,
    ...     alpha=1.5,
    ...     decay_rate=0.99,
    ...     max_vertices=100,
    ...     repository_fraction=0.5,
    ...     distance_measure="manhattan"
    ... )

    >>> for _ in range(20):
    ...     repeated.learn_one({0: 1.0, 1: 1.0})

    >>> len(repeated.vertices)
    6

    >>> sorted(repeated.micro_clusters)
    [0]

    >>> repeated.micro_clusters[0].count
    19

    >>> euclidean = cluster.RepStream(
    ...     k=5,
    ...     alpha=1.5,
    ...     decay_rate=0.99,
    ...     max_vertices=100,
    ...     repository_fraction=0.5,
    ...     distance_measure="euclidean"
    ... )

    >>> for x, _ in stream.iter_array(X):
    ...     euclidean.learn_one(x)

    >>> euclidean.n_clusters
    2

    """

    def __init__(
        self,
        k: int = 5,
        alpha: float = 1.5,
        decay_rate: float = 0.99,
        max_vertices: int = 100,
        repository_fraction: float = 0.5,
        distance_measure: str = "euclidean",
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.max_vertices = max_vertices
        self.repository_fraction = repository_fraction
        self.distance_measure = distance_measure

        self._measure = _MEASURES[distance_measure]
        self._log_decay = math.log(decay_rate)
        self._budget = max(1, max_vertices)
        self._repository_capacity = max(
            0, min(int(self._budget * repository_fraction), self._budget - 1)
        )

        self._time_stamp = 0
        self._next_index = 0
        self._next_cluster = 0
        self._keys: list = []
        self._positions: dict = {}
        self._vertices: dict[int, RepStreamVertex] = {}
        self._representatives: set[int] = set()
        self._singularities: set[int] = set()
        self._clusters: dict[int, set[int]] = {}
        self._repository: dict[int, float] = {}
        self._repository_heap: list = []
        self._deletion_queue: collections.deque = collections.deque()
        self._pending_merges: list = []
        self._pending_splits: list = []

    def _vectorize(self, x):
        positions = self._positions
        keys = self._keys
        fresh = [k for k in x if k not in positions]
        if fresh:
            try:
                fresh.sort()
            except TypeError:
                fresh.sort(key=repr)
            for k in fresh:
                positions[k] = len(keys)
                keys.append(k)
            width = len(keys)
            for vertex in self._vertices.values():
                values = vertex.values
                values.extend([0.0] * (width - len(values)))
        return [float(x.get(k, 0.0)) for k in keys]

    def _project(self, x):
        return [float(x.get(k, 0.0)) for k in self._keys]

    def _as_dict(self, values):
        return dict(zip(self._keys, values))

    @staticmethod
    def _furthest(edges):
        return max(edges.items(), key=lambda item: (item[1], item[0]))

    def _nearest(self, values, count, exclude=None):
        measure = self._measure
        candidates = [
            (measure(values, vertex.values), i)
            for i, vertex in self._vertices.items()
            if i != exclude
        ]
        if len(candidates) <= count:
            candidates.sort()
            return candidates
        return heapq.nsmallest(count, candidates)

    def _nearest_representatives(self, values, count, exclude=None):
        vertices = self._vertices
        measure = self._measure
        candidates = [
            (measure(values, vertices[i].values), i) for i in self._representatives if i != exclude
        ]
        if len(candidates) <= count:
            candidates.sort()
            return candidates
        return heapq.nsmallest(count, candidates)

    def _nearest_connected_representative(self, vertex):
        vertices = self._vertices
        in_edges = vertex.in_edges
        best = None
        for j, distance in vertex.out.items():
            if j not in in_edges:
                continue
            other = vertices.get(j)
            if other is None or not other.is_representative:
                continue
            if best is None or (distance, j) < best:
                best = (distance, j)
        if best is None:
            return None
        return vertices[best[1]]

    def _update_density(self, vertex):
        if vertex.is_singularity:
            vertex.density = 0.0
            return
        out = vertex.out
        if not out:
            vertex.density = 0.0
            return
        total = math.fsum(out.values())
        vertex.density = total / len(out)
        if total == 0.0 and len(out) == self.k:
            vertex.is_singularity = True
            vertex.density = 0.0
            self._singularities.add(vertex.index)

    def _usefulness(self, vertex, count):
        return self._log_decay * (self._time_stamp - vertex.created_at + 1) + math.log(count + 1)

    def _repository_key(self, vertex):
        return math.log(vertex.count + 1) - self._log_decay * vertex.created_at

    def _repository_push(self, vertex):
        key = self._repository_key(vertex)
        self._repository[vertex.index] = key
        heapq.heappush(self._repository_heap, (key, vertex.index))

    def _repository_remove(self, index):
        self._repository.pop(index, None)

    def _repository_first(self):
        heap = self._repository_heap
        repository = self._repository
        while heap:
            key, index = heap[0]
            if repository.get(index) == key:
                return index
            heapq.heappop(heap)
        return None

    def _repository_admit(self, vertex):
        if vertex.index in self._repository:
            return
        if len(self._repository) < self._repository_capacity:
            self._repository_push(vertex)

    def _update_repository(self, vertex, new_count):
        deleted = None
        index = vertex.index
        member = index in self._repository
        vertex.count = new_count
        if member:
            self._repository_push(vertex)
        elif len(self._repository) < self._repository_capacity:
            self._repository_push(vertex)
        else:
            worst = self._repository_first()
            if worst is not None and self._repository_key(
                self._vertices[worst]
            ) <= self._repository_key(vertex):
                self._repository_remove(worst)
                self._repository_push(vertex)
                deleted = worst
        return deleted

    def _queue_merge(self, left, right):
        self._pending_merges.append((left, right))

    def _queue_split(self, index):
        self._pending_splits.append(index)

    def _move_vertex(self, vertex, cluster):
        members = self._clusters.get(vertex.cluster)
        if members is not None:
            members.discard(vertex.index)
            if not members:
                del self._clusters[vertex.cluster]
        vertex.cluster = cluster
        self._clusters.setdefault(cluster, set()).add(vertex.index)

    def _merge_clusters(self, left, right):
        clusters = self._clusters
        if left == right or left not in clusters or right not in clusters:
            return
        if (len(clusters[right]), -right) > (len(clusters[left]), -left):
            left, right = right, left
        members = clusters[right]
        for i in members:
            self._vertices[i].cluster = left
        clusters[left] |= members
        del clusters[right]

    def _split_check(self, cluster):
        members = self._clusters.get(cluster)
        if not members:
            return
        original = sorted(members)
        vertices = self._vertices
        reps = [i for i in original if vertices[i].is_representative]
        if len(reps) <= 1:
            return
        unseen = set(reps)
        components = []
        for start in reps:
            if start not in unseen:
                continue
            unseen.discard(start)
            stack = [start]
            component = []
            while stack:
                i = stack.pop()
                component.append(i)
                for j in vertices[i].dr:
                    if j in unseen:
                        unseen.discard(j)
                        stack.append(j)
            components.append(component)
        if len(components) <= 1:
            return
        components.sort(key=lambda component: (-len(component), min(component)))
        for component in components[1:]:
            target = self._next_cluster
            self._next_cluster += 1
            for i in component:
                self._move_vertex(vertices[i], target)
        for i in original:
            vertex = vertices.get(i)
            if vertex is None or vertex.is_representative:
                continue
            rep = self._nearest_connected_representative(vertex)
            if rep is None:
                continue
            vertex.nearest_rep = rep.index
            vertex.nearest_rep_distance = vertex.out[rep.index]
            if vertex.cluster != rep.cluster:
                self._move_vertex(vertex, rep.cluster)

    def _drain_merges(self):
        vertices = self._vertices
        pending = self._pending_merges
        while pending:
            left, right = pending.pop(0)
            source = vertices.get(left)
            target = vertices.get(right)
            if source is None or target is None:
                continue
            if right not in source.dr:
                continue
            self._merge_clusters(source.cluster, target.cluster)

    def _drain_splits(self):
        vertices = self._vertices
        pending = self._pending_splits
        checked = set()
        while pending:
            index = pending.pop(0)
            vertex = vertices.get(index)
            if vertex is None or not vertex.is_representative:
                continue
            if vertex.cluster in checked:
                continue
            checked.add(vertex.cluster)
            self._split_check(vertex.cluster)

    def _update_representative_status(self, vertex):
        k = self.k
        alpha = self.alpha
        index = vertex.index
        vertices = self._vertices
        if not vertex.is_exemplar and 2 * len(vertex.rep_in) >= k:
            vertex.is_exemplar = True
        changed = False
        for j, distance in list(vertex.rep_out.items()):
            other = vertices.get(j)
            if other is None:
                continue
            blocked = vertex.is_singularity or other.is_singularity
            related = distance <= vertex.density * alpha and distance <= other.density * alpha
            if j in vertex.dr:
                if blocked or not related:
                    vertex.dr.discard(j)
                    other.dr.discard(index)
                    self._queue_split(index)
                    changed = True
            elif index in other.rep_out and not blocked and related:
                vertex.dr.add(j)
                other.dr.add(index)
                self._queue_merge(index, j)
                changed = True
        return changed

    def _make_representative(self, vertex):
        index = vertex.index
        vertex.is_representative = True
        vertex.is_exemplar = 2 * len(vertex.rep_in) >= self.k
        self._representatives.add(index)
        self._update_density(vertex)
        if vertex.cluster is None:
            cluster = self._next_cluster
            self._next_cluster += 1
            vertex.cluster = cluster
            self._clusters[cluster] = {index}
        neighbours = self._nearest_representatives(vertex.values, self.k, exclude=index)
        self._link_into_rsg(vertex, neighbours)
        vertices = self._vertices
        for j, distance in list(vertex.out.items()):
            other = vertices.get(j)
            if other is None or other.is_representative:
                continue
            current = other.nearest_rep
            if current is None or current not in vertices:
                closer = True
            else:
                closer = (distance, index) < (other.nearest_rep_distance, current)
            if closer:
                other.nearest_rep = index
                other.nearest_rep_distance = distance
                if other.cluster != vertex.cluster:
                    self._move_vertex(other, vertex.cluster)
        self._repository_admit(vertex)

    def _link_into_rsg(self, vertex, neighbours):
        index = vertex.index
        vertices = self._vertices
        k = self.k
        created = []
        for distance, j in neighbours:
            other = vertices[j]
            vertex.rep_out[j] = distance
            other.rep_in.add(index)
            if len(other.rep_out) < k or distance <= self._furthest(other.rep_out)[1]:
                other.rep_out[index] = distance
                if len(other.rep_out) > k:
                    far_index = self._furthest(other.rep_out)[0]
                    del other.rep_out[far_index]
                    other.dr.discard(far_index)
                    far = vertices.get(far_index)
                    if far is not None:
                        far.rep_in.discard(j)
                        if far_index != index and j in far.dr:
                            far.dr.discard(j)
                            self._queue_split(j)
                            self._queue_split(far_index)
                if index in other.rep_out:
                    vertex.rep_in.add(j)
                    created.append(j)
        self._update_representative_status(vertex)
        for j in created:
            other = vertices.get(j)
            if other is not None:
                self._update_representative_status(other)

    def _link_into_sg(self, vertex, neighbours):
        index = vertex.index
        vertices = self._vertices
        k = self.k
        created = []
        removed_recip = []
        removed_neighbours = []
        for distance, j in neighbours:
            other = vertices[j]
            vertex.out[j] = distance
            other.in_edges.add(index)
            if len(other.out) < k or distance <= self._furthest(other.out)[1]:
                other.out[index] = distance
                if len(other.out) > k:
                    far_index = self._furthest(other.out)[0]
                    del other.out[far_index]
                    removed_neighbours.append((j, far_index))
                    far = vertices.get(far_index)
                    if far is not None:
                        far.in_edges.discard(j)
                        if far_index != index and j in far.out:
                            removed_recip.append(j)
                if index in other.out:
                    vertex.in_edges.add(j)
                    created.append(j)

        retired: set[int] = set()
        for j in created:
            if j in retired:
                continue
            other = vertices.get(j)
            if other is None or not other.is_representative:
                continue
            self._update_density(other)
            deleted = self._update_repository(other, other.count + 1)
            if deleted is not None:
                retired.add(deleted)
                self._unlink_vertex(deleted)
            if j in vertices:
                self._update_representative_status(other)

        rep = self._nearest_connected_representative(vertex)
        if rep is None:
            self._make_representative(vertex)
        else:
            vertex.nearest_rep = rep.index
            vertex.nearest_rep_distance = vertex.out[rep.index]
            self._move_vertex(vertex, rep.cluster)

        for j in removed_recip:
            if j in retired:
                continue
            other = vertices.get(j)
            if other is None or not other.is_representative:
                continue
            self._update_density(other)
            self._update_representative_status(other)

        for j, far_index in removed_neighbours:
            other = vertices.get(j)
            candidates = [j, far_index]
            if other is not None:
                candidates.extend(sorted(other.out))
            for m in candidates:
                if m in retired:
                    continue
                neighbour = vertices.get(m)
                if neighbour is None or neighbour.is_representative:
                    continue
                if self._nearest_connected_representative(neighbour) is None:
                    self._make_representative(neighbour)

    def _refill_sg(self, vertex):
        missing = self.k - len(vertex.out)
        if missing <= 0:
            return
        index = vertex.index
        out = vertex.out
        values = vertex.values
        vertices = self._vertices
        measure = self._measure
        if missing == 1:
            best_index = -1
            best_distance = math.inf
            for i, other in vertices.items():
                if i == index or i in out:
                    continue
                distance = measure(values, other.values)
                if distance < best_distance:
                    best_distance = distance
                    best_index = i
            found = () if best_index < 0 else ((best_distance, best_index),)
        else:
            found = heapq.nsmallest(
                missing,
                (
                    (measure(values, other.values), i)
                    for i, other in vertices.items()
                    if i != index and i not in out
                ),
            )
        for distance, i in found:
            out[i] = distance
            vertices[i].in_edges.add(index)

    def _refill_rsg(self, vertex):
        missing = self.k - len(vertex.rep_out)
        if missing <= 0:
            return
        index = vertex.index
        rep_out = vertex.rep_out
        values = vertex.values
        vertices = self._vertices
        measure = self._measure
        found = heapq.nsmallest(
            missing,
            (
                (measure(values, vertices[i].values), i)
                for i in sorted(self._representatives)
                if i != index and i not in rep_out
            ),
        )
        for distance, i in found:
            rep_out[i] = distance
            vertices[i].rep_in.add(index)

    def _unlink_vertex(self, index):
        vertices = self._vertices
        vertex = vertices.pop(index, None)
        if vertex is None:
            return
        was_representative = vertex.is_representative
        self._repository_remove(index)
        self._representatives.discard(index)
        self._singularities.discard(index)
        members = self._clusters.get(vertex.cluster)
        if members is not None:
            members.discard(index)
            if not members:
                del self._clusters[vertex.cluster]

        for j in vertex.out:
            other = vertices.get(j)
            if other is not None:
                other.in_edges.discard(index)
        affected = []
        for j in vertex.in_edges:
            other = vertices.get(j)
            if other is None:
                continue
            other.out.pop(index, None)
            affected.append(j)

        rep_affected = []
        for j in vertex.rep_out:
            other = vertices.get(j)
            if other is None:
                continue
            other.rep_in.discard(index)
            if index in other.dr:
                other.dr.discard(index)
                self._queue_split(j)
        for j in vertex.rep_in:
            other = vertices.get(j)
            if other is None:
                continue
            other.rep_out.pop(index, None)
            if index in other.dr:
                other.dr.discard(index)
                self._queue_split(j)
            rep_affected.append(j)

        affected.sort()
        rep_affected.sort()

        for j in affected:
            other = vertices.get(j)
            if other is not None:
                self._refill_sg(other)
        for j in affected:
            other = vertices.get(j)
            if other is not None and other.is_representative:
                self._update_density(other)
                self._update_representative_status(other)
        for j in rep_affected:
            other = vertices.get(j)
            if other is not None and other.is_representative:
                self._refill_rsg(other)
                self._update_representative_status(other)

        if was_representative:
            for other in vertices.values():
                if other.nearest_rep == index:
                    other.nearest_rep = None
                    other.nearest_rep_distance = math.inf

        for j in affected:
            other = vertices.get(j)
            if other is None or other.is_representative:
                continue
            rep = self._nearest_connected_representative(other)
            if rep is None:
                self._make_representative(other)
            else:
                other.nearest_rep = rep.index
                other.nearest_rep_distance = other.out[rep.index]
                if other.cluster != rep.cluster:
                    self._move_vertex(other, rep.cluster)

    def _matching_singularity(self, values):
        for i in sorted(self._singularities):
            vertex = self._vertices.get(i)
            if vertex is not None and vertex.values == values:
                return vertex
        return None

    def _next_retirement(self):
        queue = self._deletion_queue
        while queue:
            index = queue.popleft()
            if index in self._vertices and index not in self._repository:
                return index
        return None

    def _enforce_budget(self):
        while len(self._vertices) > self._budget:
            index = self._next_retirement()
            if index is None:
                return
            self._unlink_vertex(index)
            self._drain_merges()
            self._drain_splits()

    def learn_one(self, x, w=None):
        values = self._vectorize(x)

        singularity = self._matching_singularity(values)
        if singularity is not None:
            deleted = self._update_repository(singularity, singularity.count + 1)
            if deleted is not None:
                self._unlink_vertex(deleted)
                self._drain_merges()
                self._drain_splits()
            self._time_stamp += 1
            return

        neighbours = self._nearest(values, self.k)
        index = self._next_index
        self._next_index += 1
        vertex = RepStreamVertex(values=values, index=index, created_at=self._time_stamp)
        self._vertices[index] = vertex
        self._deletion_queue.append(index)

        self._link_into_sg(vertex, neighbours)
        self._drain_merges()
        self._drain_splits()
        self._enforce_budget()
        self._time_stamp += 1

    def predict_one(self, x, w=None):
        values = self._project(x)
        vertices = self._vertices
        measure = self._measure
        best = None
        for i in self._representatives:
            distance = measure(values, vertices[i].values)
            if best is None or (distance, i) < best:
                best = (distance, i)
        if best is None:
            return 0
        return vertices[best[1]].cluster

    @property
    def n_clusters(self) -> int:
        return len(self._clusters)

    @property
    def clusters(self) -> dict[int, set[int]]:
        return self._clusters

    @property
    def vertices(self) -> dict[int, RepStreamVertex]:
        return self._vertices

    @property
    def micro_clusters(self) -> dict[int, RepStreamVertex]:
        return {i: self._vertices[i] for i in sorted(self._representatives)}

    @property
    def exemplars(self) -> dict[int, RepStreamVertex]:
        return {
            i: self._vertices[i]
            for i in sorted(self._representatives)
            if self._vertices[i].is_exemplar
        }

    @property
    def predictors(self) -> dict[int, RepStreamVertex]:
        return {
            i: self._vertices[i]
            for i in sorted(self._representatives)
            if not self._vertices[i].is_exemplar
        }

    @property
    def repository(self) -> dict[int, RepStreamVertex]:
        return {i: self._vertices[i] for i in sorted(self._repository)}

    @property
    def centers(self) -> dict:
        centers = {}
        for cluster, members in self._clusters.items():
            points = [
                self._vertices[i].values for i in sorted(members) if i in self._representatives
            ]
            if not points:
                points = [self._vertices[i].values for i in sorted(members)]
            width = len(self._keys)
            total = [0.0] * width
            for point in points:
                for j in range(width):
                    total[j] += point[j]
            centers[cluster] = self._as_dict([value / len(points) for value in total])
        return centers


def _euclidean(point_a, point_b):
    return math.dist(point_a, point_b)


def _manhattan(point_a, point_b):
    return sum(map(abs, map(operator.sub, point_a, point_b)))


_MEASURES = {"euclidean": _euclidean, "manhattan": _manhattan}


class RepStreamVertex:
    __slots__ = (
        "cluster",
        "count",
        "created_at",
        "density",
        "dr",
        "in_edges",
        "index",
        "is_exemplar",
        "is_representative",
        "is_singularity",
        "nearest_rep",
        "nearest_rep_distance",
        "out",
        "rep_in",
        "rep_out",
        "values",
    )

    def __init__(self, values, index, created_at):
        self.values = list(values)
        self.index = index
        self.created_at = created_at
        self.count = 0
        self.density = 0.0
        self.cluster = None
        self.nearest_rep = None
        self.nearest_rep_distance = math.inf
        self.is_representative = False
        self.is_exemplar = False
        self.is_singularity = False
        self.out: dict[int, float] = {}
        self.in_edges: set[int] = set()
        self.rep_out: dict[int, float] = {}
        self.rep_in: set[int] = set()
        self.dr: set[int] = set()
