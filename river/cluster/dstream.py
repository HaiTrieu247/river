from __future__ import annotations

import math
from collections import deque

from river import base
from river.utils.vectordict import euclidean_distance_dict

__all__ = ["DStream"]

_SPARSE = 0
_TRANSITIONAL = 1
_DENSE = 2


class DStream(base.Clusterer):
    r"""D-Stream

    D-Stream [^1] is a density-based clustering algorithm for evolving data streams. It does
    not require the number of clusters to be known in advance, it discovers clusters of
    arbitrary shape, and it discards outliers.

    The data space is partitioned into a grid of hyper-rectangular cells of side
    `grid_width`. Only the cells that actually receive data are kept in memory, in a hash
    table called the grid list.

    **Online grid maintenance (learning)**

    Every incoming record is mapped to the grid that contains it, and that grid's density is
    updated. Each record carries a density coefficient that decays as $\lambda^{t - t_c}$,
    where $t_c$ is its arrival time. Rather than decaying every grid at every step, a grid's
    density is refreshed only when it receives a record:

    $$D(g, t_n) = \lambda^{t_n - t_l} D(g, t_l) + 1$$

    where $t_l$ is the last time the grid was updated. This makes the online phase run in
    constant time per record.

    **Offline clustering**

    Every `gap` steps the grids are re-inspected and the clusters are adjusted. Since the
    total density in the system converges to $\frac{1}{1 - \lambda}$, and there are `n_grids`
    grids in the partitioned space, the average grid density approaches
    $\frac{1}{n\_grids (1 - \lambda)}$. A grid is called *dense* when its density reaches
    $C_m$ times that average, *sparse* when it falls below $C_l$ times that average, and
    *transitional* in between. A cluster is a connected group of grids whose interior is
    dense and whose border is dense or transitional.

    The first inspection builds the clusters from scratch by growing connected components out
    of the dense grids. Later inspections only look at the grids whose density class changed,
    and attach, detach, merge or split clusters accordingly.

    The incremental procedure printed in the paper only ever looks at a newly dense grid's
    largest neighbouring cluster, so it leaves two clusters separate even once that grid
    bridges them, and it leaves the grid unlabelled when that neighbour turns out to be a
    transitional one it may not join. This implementation goes on to merge every neighbouring
    dense cluster and to open a cluster for a dense grid that ends up with none, so that an
    inspection reaches the clusters the initial pass would have built.

    **Sporadic grid removal**

    Grids mapped to by outliers would otherwise accumulate forever. A sparse grid is judged
    *sporadic* when its density falls below the threshold function

    $$\pi(t_g, t) = \frac{C_l (1 - \lambda^{t - t_g + 1})}{n\_grids (1 - \lambda)}$$

    and enough time has passed since it was last removed, namely $t \geq (1 + \beta) t_m$.
    Sporadic grids are first marked, then deleted at the following inspection if they have
    received no data in the meantime. The paper shows that a grid removed this way could not
    have become transitional or dense had its history been kept, so the removal does not
    degrade the clustering.

    A deleted grid's $t_m$ outlives its characteristic vector, otherwise the guard above would
    never fire on a grid that keeps coming back. Those timestamps are dropped again as soon as
    $t \geq (1 + \beta) t_m$ holds for good, so only the grids deleted over the last
    $\beta t$ steps are remembered.

    Parameters
    ----------
    grid_width
        Side length of a grid cell along every dimension. Data is expected to be normalised,
        typically to `[0, 1]`.
    decaying_factor
        The decay factor $\lambda$, within the range `(0, 1)`. Values close to `1` make the
        model retain the past for longer.
    dense_threshold
        The $C_m$ coefficient of the dense-grid threshold. Must be greater than `1` and
        smaller than `n_grids`.
    sparse_threshold
        The $C_l$ coefficient of the sparse-grid threshold. Must be within the range `(0, 1)`
        and smaller than `dense_threshold`.
    beta
        The $\beta$ coefficient guarding the deletion of a grid that has already been deleted
        before. Must be greater than `0`.
    n_grids
        The number of grids $N$ the data space is partitioned into. It only enters the
        algorithm through the density thresholds, which are all proportional to
        $\frac{1}{n\_grids (1 - \lambda)}$. The theoretical value $N = \prod_i p_i$ is
        unusable in high dimension, since it makes the thresholds vanish, so it is left as a
        parameter to be tuned.
    gap
        Number of steps between two offline inspections. When left to `None` it is derived
        from the other parameters, as the smallest number of steps in which a dense grid can
        decay into a sparse one or a sparse grid can grow into a dense one. That second bound
        collapses towards `0` as `n_grids` grows, in which case the derived value is clamped
        to `1` and the offline phase runs at every step. Setting `gap` by hand is worthwhile
        then, since the offline phase is linear in the number of grids held in memory.

    Attributes
    ----------
    n_clusters
        Number of clusters generated by the algorithm.
    clusters
        The grids of each cluster, keyed by cluster label. A grid is a tuple of
        `(feature, coordinate)` pairs, where coordinates equal to `0` are omitted.
    centers
        The density-weighted centre of each cluster.
    grids
        The characteristic vector of every grid currently held in memory.

    References
    ----------
    [^1]: Chen, Y. and Tu, L. (2007, pp 133-142). Density-Based Clustering for Real-Time
          Stream Data. In Proceedings of the 13th ACM SIGKDD International Conference on
          Knowledge Discovery and Data Mining, August 12-15, 2007, San Jose, CA, USA.

    Examples
    --------

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 0.5], [1, 0.625], [1, 0.75], [1, 1.125], [1, 1.5], [1, 1.75],
    ...     [4, 1.5], [4, 2.25], [4, 2.5], [4, 3], [4, 3.25], [4, 3.5]
    ... ]

    >>> dstream = cluster.DStream(
    ...     grid_width=1.0,
    ...     decaying_factor=0.998,
    ...     dense_threshold=2.5,
    ...     sparse_threshold=0.8,
    ...     n_grids=500,
    ... )

    >>> for x, _ in stream.iter_array(X):
    ...     dstream.learn_one(x)

    >>> dstream.n_clusters
    2

    >>> dstream.predict_one({0: 1, 1: 1})
    0

    >>> dstream.predict_one({0: 4, 1: 3})
    1

    """

    def __init__(
        self,
        grid_width: float = 1.0,
        decaying_factor: float = 0.998,
        dense_threshold: float = 3.0,
        sparse_threshold: float = 0.8,
        beta: float = 0.3,
        n_grids: int = 100,
        gap: int | None = None,
    ):
        super().__init__()

        if grid_width <= 0:
            raise ValueError(
                f"The value of `grid_width` (currently {grid_width}) must be greater than 0."
            )
        if not 0 < decaying_factor < 1:
            raise ValueError(
                f"The value of `decaying_factor` (currently {decaying_factor}) must be within "
                "the range (0, 1)."
            )
        if dense_threshold <= 1:
            raise ValueError(
                f"The value of `dense_threshold` (currently {dense_threshold}) must be greater "
                "than 1."
            )
        if not 0 < sparse_threshold < 1:
            raise ValueError(
                f"The value of `sparse_threshold` (currently {sparse_threshold}) must be within "
                "the range (0, 1)."
            )
        if beta <= 0:
            raise ValueError(f"The value of `beta` (currently {beta}) must be greater than 0.")
        if n_grids <= dense_threshold:
            raise ValueError(
                f"The value of `n_grids` (currently {n_grids}) must be greater than "
                f"`dense_threshold` (currently {dense_threshold})."
            )
        if gap is not None and gap < 1:
            raise ValueError(f"The value of `gap` (currently {gap}) must be greater than 0.")

        self.grid_width = grid_width
        self.decaying_factor = decaying_factor
        self.dense_threshold = dense_threshold
        self.sparse_threshold = sparse_threshold
        self.beta = beta
        self.n_grids = n_grids
        self.gap = gap

        scale = n_grids * (1 - decaying_factor)
        self._dense_density = dense_threshold / scale
        self._sparse_density = sparse_threshold / scale
        self._gap = gap if gap is not None else self._compute_gap()

        self._time = 0
        self._last_inspection = 0
        self._initialized = False
        self._next_label = 0
        self._features: set = set()
        self._grids: dict[tuple, DStreamGrid] = {}
        self._removals: dict[tuple, int] = {}
        self._clusters: dict[int, set[tuple]] = {}
        self._centers: dict[int, dict] = {}
        self._dirty = False
        self._stale_centers = False

    def _compute_gap(self) -> int:
        ratio = max(
            self.sparse_threshold / self.dense_threshold,
            (self.n_grids - self.dense_threshold) / (self.n_grids - self.sparse_threshold),
        )
        return max(1, int(math.log(ratio) / math.log(self.decaying_factor)))

    @property
    def n_clusters(self) -> int:
        return len(self._clusters)

    @property
    def clusters(self) -> dict[int, set[tuple]]:
        return self._clusters

    @property
    def centers(self) -> dict[int, dict]:
        if self._stale_centers:
            self._update_centers()
            self._stale_centers = False
        return self._centers

    @property
    def grids(self) -> dict[tuple, DStreamGrid]:
        return self._grids

    def _grid_of(self, x) -> tuple:
        width = self.grid_width
        coordinates = []
        for feature, value in x.items():
            coordinate = int(value // width)
            if coordinate:
                coordinates.append((feature, coordinate))
        coordinates.sort()
        return tuple(coordinates)

    @staticmethod
    def _shift(key, feature, coordinate) -> tuple:
        if coordinate:
            shifted = (feature, coordinate)
            for i, (other, _) in enumerate(key):
                if other == feature:
                    return key[:i] + (shifted,) + key[i + 1 :]
                if other > feature:
                    return key[:i] + (shifted,) + key[i:]
            return key + (shifted,)
        for i, (other, _) in enumerate(key):
            if other == feature:
                return key[:i] + key[i + 1 :]
        return key

    def _neighbours(self, key):
        coordinates = dict(key)
        for feature in self._features:
            centre = coordinates.get(feature, 0)
            yield self._shift(key, feature, centre - 1)
            yield self._shift(key, feature, centre + 1)

    def _is_inside(self, key, members) -> bool:
        for neighbour in self._neighbours(key):
            if neighbour not in members:
                return False
        return bool(self._features)

    def _density(self, grid) -> float:
        return grid.density * self.decaying_factor ** (self._time - grid.last_update)

    def _attribute(self, density) -> int:
        if density >= self._dense_density:
            return _DENSE
        if density <= self._sparse_density:
            return _SPARSE
        return _TRANSITIONAL

    def _threshold(self, last_update) -> float:
        return (
            self.sparse_threshold
            * (1 - self.decaying_factor ** (self._time - last_update + 1))
            / (self.n_grids * (1 - self.decaying_factor))
        )

    def learn_one(self, x, w=1.0):
        self._features.update(x)
        key = self._grid_of(x)
        grid = self._grids.get(key)
        if grid is None:
            grid = DStreamGrid(self._time, self._removals.get(key, 0))
            self._grids[key] = grid
        grid.density = grid.density * self.decaying_factor ** (self._time - grid.last_update) + w
        grid.last_update = self._time

        if not self._initialized:
            if self._time >= self._gap:
                self._initial_clustering()
                self._initialized = True
                self._last_inspection = self._time
        elif self._time % self._gap == 0:
            self._remove_sporadic_grids()
            self._adjust_clustering()
            self._last_inspection = self._time

        self._time += 1

    def predict_one(self, x, w=None):
        grid = self._grids.get(self._grid_of(x))
        if grid is not None and grid.label is not None:
            return grid.label
        closest = 0
        shortest = math.inf
        for label, center in self.centers.items():
            distance = euclidean_distance_dict(center, x)
            if distance < shortest:
                shortest = distance
                closest = label
        return closest

    def _new_cluster(self, key, grid) -> int:
        label = self._next_label
        self._next_label += 1
        self._clusters[label] = {key}
        grid.label = label
        self._dirty = True
        return label

    def _attach(self, key, grid, label):
        self._clusters[label].add(key)
        grid.label = label
        self._dirty = True

    def _detach(self, key, label):
        members = self._clusters.get(label)
        if members is None:
            return
        members.discard(key)
        self._dirty = True
        if not members:
            del self._clusters[label]
        else:
            self._split(label)

    def _merge(self, source, target):
        members = self._clusters.pop(source)
        for key in members:
            self._grids[key].label = target
        self._clusters[target].update(members)
        self._dirty = True

    def _split(self, label):
        members = self._clusters[label]
        components = []
        unvisited = set(members)
        while unvisited:
            component = {unvisited.pop()}
            queue = deque(component)
            while queue:
                for neighbour in self._neighbours(queue.popleft()):
                    if neighbour in unvisited:
                        unvisited.discard(neighbour)
                        component.add(neighbour)
                        queue.append(neighbour)
            components.append(component)
        if len(components) == 1:
            return
        self._clusters[label] = components[0]
        for component in components[1:]:
            new_label = self._next_label
            self._next_label += 1
            self._clusters[new_label] = component
            for key in component:
                self._grids[key].label = new_label

    def _initial_clustering(self):
        for grid in self._grids.values():
            grid.attribute = self._attribute(self._density(grid))
            grid.label = None
        self._clusters = {}
        self._next_label = 0

        visited: set = set()
        for key, grid in self._grids.items():
            if grid.attribute != _DENSE or key in visited:
                continue
            label = self._next_label
            self._next_label += 1
            members: set = set()
            visited.add(key)
            queue = deque([key])
            while queue:
                current = queue.popleft()
                members.add(current)
                self._grids[current].label = label
                for neighbour in self._neighbours(current):
                    other = self._grids.get(neighbour)
                    if other is None or neighbour in visited or other.attribute == _SPARSE:
                        continue
                    visited.add(neighbour)
                    queue.append(neighbour)
            self._clusters[label] = members

        self._relabel()

    def _remove_sporadic_grids(self):
        now = self._time
        deleted = []
        for key, grid in self._grids.items():
            sporadic = (
                self._density(grid) < self._threshold(grid.last_update)
                and now >= (1 + self.beta) * grid.last_removal
            )
            if grid.sporadic:
                if grid.last_update <= self._last_inspection:
                    deleted.append(key)
                elif not sporadic:
                    grid.sporadic = False
            elif sporadic:
                grid.sporadic = True

        for key in deleted:
            grid = self._grids.pop(key)
            if grid.label is not None:
                self._detach(key, grid.label)
            self._removals[key] = now

        for key in [k for k, t in self._removals.items() if now >= (1 + self.beta) * t]:
            del self._removals[key]

    def _adjust_clustering(self):
        changed = []
        for key, grid in self._grids.items():
            attribute = self._attribute(self._density(grid))
            if attribute != grid.attribute:
                grid.attribute = attribute
                changed.append(key)

        for key in changed:
            grid = self._grids.get(key)
            if grid is None:
                continue
            if grid.attribute == _SPARSE:
                if grid.label is not None:
                    label = grid.label
                    grid.label = None
                    self._detach(key, label)
            elif grid.attribute == _DENSE:
                self._adjust_dense(key, grid)
            else:
                self._adjust_transitional(key, grid)

        if self._dirty:
            self._relabel()

    def _largest_neighbouring_cluster(self, key):
        best = None
        best_size = -1
        for neighbour in self._neighbours(key):
            other = self._grids.get(neighbour)
            if other is None or other.label is None:
                continue
            size = len(self._clusters[other.label])
            if size > best_size:
                best_size = size
                best = (neighbour, other)
        return best, best_size

    def _adjust_dense(self, key, grid):
        best, best_size = self._largest_neighbouring_cluster(key)
        if best is None:
            if grid.label is None:
                self._new_cluster(key, grid)
            return

        neighbour, other = best
        host = other.label

        if other.attribute == _DENSE:
            if grid.label is None:
                self._attach(key, grid, host)
            elif grid.label != host:
                own = grid.label
                if len(self._clusters[own]) > best_size:
                    self._merge(host, own)
                else:
                    self._merge(own, host)
        else:
            if grid.label is None:
                if not self._is_inside(neighbour, self._clusters[host] | {key}):
                    self._attach(key, grid, host)
            elif grid.label != host and len(self._clusters[grid.label]) >= best_size:
                own = grid.label
                other.label = own
                self._clusters[own].add(neighbour)
                self._detach(neighbour, host)

        if grid.label is None:
            self._new_cluster(key, grid)

        self._absorb_dense_neighbours(key, grid)

    def _absorb_dense_neighbours(self, key, grid):
        if grid.label is None:
            return
        for neighbour in self._neighbours(key):
            other = self._grids.get(neighbour)
            if other is None or other.attribute != _DENSE or other.label is None:
                continue
            own, host = grid.label, other.label
            if own == host:
                continue
            if len(self._clusters[own]) >= len(self._clusters[host]):
                self._merge(host, own)
            else:
                self._merge(own, host)

    def _adjust_transitional(self, key, grid):
        best_label = None
        best_size = -1
        inspected: set = set()
        for neighbour in self._neighbours(key):
            other = self._grids.get(neighbour)
            if other is None or other.label is None or other.label in inspected:
                continue
            inspected.add(other.label)
            members = self._clusters[other.label]
            if len(members) > best_size and not self._is_inside(key, members | {key}):
                best_size = len(members)
                best_label = other.label

        if best_label is None or best_label == grid.label:
            return
        if grid.label is not None:
            label = grid.label
            grid.label = None
            self._detach(key, label)
        self._attach(key, grid, best_label)

    def _relabel(self):
        clusters = {}
        for label, old in enumerate(sorted(self._clusters)):
            members = self._clusters[old]
            clusters[label] = members
            for key in members:
                self._grids[key].label = label
        self._clusters = clusters
        self._next_label = len(clusters)
        self._dirty = False
        self._stale_centers = True

    def _update_centers(self):
        half = self.grid_width / 2
        centers = {}
        for label, members in self._clusters.items():
            total = 0.0
            center: dict = {}
            for key in members:
                grid = self._grids.get(key)
                if grid is None:
                    continue
                weight = self._density(grid)
                if weight <= 0:
                    continue
                total += weight
                coordinates = dict(key)
                for feature in self._features:
                    center[feature] = center.get(feature, 0.0) + weight * (
                        coordinates.get(feature, 0) * self.grid_width + half
                    )
            if total:
                centers[label] = {feature: value / total for feature, value in center.items()}
        self._centers = centers


class DStreamGrid:
    """D-Stream characteristic vector."""

    __slots__ = ("density", "last_update", "last_removal", "label", "sporadic", "attribute")

    def __init__(self, last_update, last_removal=0):
        self.density = 0.0
        self.last_update = last_update
        self.last_removal = last_removal
        self.label: int | None = None
        self.sporadic = False
        self.attribute = _SPARSE
