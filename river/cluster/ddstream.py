from __future__ import annotations

import math
from collections import deque

from river import base
from river.utils.vectordict import euclidean_distance_dict

__all__ = ["DDStream"]

_SPARSE = 0
_TRANSITIONAL = 1
_DENSE = 2


class DDStream(base.Clusterer):
    r"""DD-Stream

    DD-Stream [^1] is a grid and density-based clustering algorithm for evolving data
    streams. Like `DStream`, it partitions the data space into a grid of hyper-rectangular
    cells of side `grid_width`, keeps only the cells that receive data, and lets their
    density decay over time. What it adds is a rule for the records that land near a cell
    border, which the grid alone would file away somewhat arbitrarily.

    **Online grid maintenance (learning)**

    Every record is mapped to a grid, whose density is then refreshed:

    $$D(g, t_n) = \lambda^{t_n - t_l} D(g, t_l) + 1$$

    where $t_l$ is the last time that grid was written to. Decaying on write rather than
    decaying every grid at every step keeps the online phase constant time per record.

    **Boundary records and DCQ-means**

    A record that falls within `boundary_margin` of a cell face sits almost exactly between
    two grids, and the side it ends up on is decided by a partition the stream itself has
    outgrown. Redrawing the grid would fix that, but it costs a full pass over the data.
    DD-Stream instead keeps the grid as it is and picks the target cell with DCQ-means: among
    the containing cell and the cells across every nearby face, it takes the one whose centre
    is closest, treating two centres as equally close when the distances differ by less than
    the boundary width. Ties are settled by density first, then by which cell was written to
    most recently. In effect a border record joins the busier of the two cells it straddles,
    instead of seeding a cell of its own next to a cluster that is already there.

    Setting `boundary_margin` to `0` disables this and leaves plain grid mapping, which is
    what the rest of the algorithm assumes anyway. Following the paper, DCQ-means is also
    skipped on the steps that carry an inspection, since the grids are about to be reappraised
    anyway. A `gap` of `1` therefore turns it off altogether.

    **Offline clustering**

    Every `gap` steps the grids are re-inspected. Since the total density in the system
    converges to $\frac{1}{1 - \lambda}$ and the space holds `n_grids` grids, the average
    grid density approaches $\frac{1}{n\_grids (1 - \lambda)}$. A grid is *dense* when its
    density reaches $C_m$ times that average, *sparse* when it falls below $C_l$ times it,
    and *transitional* in between. A cluster is then grown out of a dense grid through its
    neighbours, stopping at the sparse ones, in the usual density-clustering fashion. Unlike
    `DStream`, which patches its clusters as grids change class, DD-Stream rebuilds them from
    scratch at every inspection, so cluster labels are not stable from one inspection to the
    next.

    **Forgetting grids**

    Left alone, the cells that outliers open would pile up forever. Equation (5) states that
    $D(g, t_n)$ is zero once $t_n \gg t_l$, and D-Stream [^2] gives that a workable form: a
    sparse grid is dropped once its density falls under

    $$\pi(t_g, t) = \frac{C_l (1 - \lambda^{t - t_g + 1})}{n\_grids (1 - \lambda)}$$

    where $t_g$ is the last time it was written to. The threshold starts near zero and climbs
    towards the sparse threshold as the grid sits idle, so a grid that keeps receiving data is
    given the time to build its density up, while one that stops is dropped a few inspections
    later. A grid dropped this way could not have grown past the sparse threshold had it been
    kept, so nothing is lost by letting it go and rebuilding it from zero if data comes back.

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
    boundary_margin
        How close to a cell face a record has to be, as a fraction of `grid_width`, for
        DCQ-means to consider the cell on the other side. Must be within the range
        `[0, 0.5)`. A value of `0` turns DCQ-means off.
    n_grids
        The number of grids $N$ the data space is partitioned into. It only enters the
        algorithm through the density thresholds, which are all proportional to
        $\frac{1}{n\_grids (1 - \lambda)}$. The theoretical value $N = \prod_i S_i$ is
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
        The eigenvector of every grid currently held in memory.

    References
    ----------
    [^1]: Jia, C., Tan, C. and Yong, A. (2008, pp 517-521). A Grid and Density-based
          Clustering Algorithm for Processing Data Stream. In Proceedings of the Second
          International Conference on Genetic and Evolutionary Computing, September 25-26,
          2008, Jingzhou, Hubei, China.
    [^2]: Chen, Y. and Tu, L. (2007, pp 133-142). Density-Based Clustering for Real-Time
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

    >>> ddstream = cluster.DDStream(
    ...     grid_width=1.0,
    ...     decaying_factor=0.998,
    ...     dense_threshold=2.5,
    ...     sparse_threshold=0.8,
    ...     n_grids=500,
    ... )

    >>> for x, _ in stream.iter_array(X):
    ...     ddstream.learn_one(x)

    >>> ddstream.n_clusters
    2

    >>> ddstream.predict_one({0: 1, 1: 1})
    0

    >>> ddstream.predict_one({0: 4, 1: 3})
    1

    The two records at `y = 2` below sit right on a cell face. The first one opens a cell of
    its own, whereas the second one is pulled into the cell the earlier records have filled:

    >>> ddstream = cluster.DDStream(grid_width=1.0, boundary_margin=0)
    >>> for _ in range(5):
    ...     ddstream.learn_one({0: 0.5, 1: 1.5})
    >>> ddstream.learn_one({0: 0.5, 1: 2.0})
    >>> sorted(ddstream.grids)
    [((1, 1),), ((1, 2),)]

    >>> ddstream = cluster.DDStream(grid_width=1.0, boundary_margin=0.1)
    >>> for _ in range(5):
    ...     ddstream.learn_one({0: 0.5, 1: 1.5})
    >>> ddstream.learn_one({0: 0.5, 1: 2.0})
    >>> sorted(ddstream.grids)
    [((1, 1),)]

    """

    def __init__(
        self,
        grid_width: float = 1.0,
        decaying_factor: float = 0.998,
        dense_threshold: float = 3.0,
        sparse_threshold: float = 0.8,
        boundary_margin: float = 0.1,
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
        if not 0 <= boundary_margin < 0.5:
            raise ValueError(
                f"The value of `boundary_margin` (currently {boundary_margin}) must be within "
                "the range [0, 0.5)."
            )
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
        self.boundary_margin = boundary_margin
        self.n_grids = n_grids
        self.gap = gap

        scale = n_grids * (1 - decaying_factor)
        self._dense_density = dense_threshold / scale
        self._sparse_density = sparse_threshold / scale
        self._margin = boundary_margin * grid_width
        self._gap = gap if gap is not None else self._compute_gap()

        self._time = 0
        self._known: set = set()
        self._features: list = []
        self._grids: dict[tuple, DDStreamGrid] = {}
        self._clusters: dict[int, set[tuple]] = {}
        self._centers: dict[int, dict] = {}
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
    def grids(self) -> dict[tuple, DDStreamGrid]:
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

    def _dcq_means(self, x) -> tuple:
        width = self.grid_width
        margin = self._margin
        half = width / 2

        coordinates = []
        boundary = []
        squared = 0.0
        for feature, value in x.items():
            coordinate = int(value // width)
            offset = value - coordinate * width - half
            if coordinate:
                coordinates.append((feature, coordinate))
            squared += offset * offset
            if offset < margin - half:
                boundary.append((feature, coordinate - 1, offset, offset + width))
            elif offset > half - margin:
                boundary.append((feature, coordinate + 1, offset, offset - width))

        coordinates.sort()
        key = tuple(coordinates)
        if not boundary:
            return key

        grids = self._grids
        grid = grids.get(key)
        nearest = math.sqrt(squared)
        best_key = key
        best_density = self._density(grid) if grid is not None else 0.0
        best_update = grid.last_update if grid is not None else -1

        for feature, coordinate, own, shifted in boundary:
            if math.sqrt(squared - own * own + shifted * shifted) > nearest + margin:
                continue
            candidate = self._shift(key, feature, coordinate)
            other = grids.get(candidate)
            if other is None:
                continue
            density = self._density(other)
            if density > best_density or (
                density == best_density and other.last_update > best_update
            ):
                best_key = candidate
                best_density = density
                best_update = other.last_update

        return best_key

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
        size = len(key)
        index = 0
        for feature in self._features:
            while index < size and key[index][0] < feature:
                index += 1
            if index < size and key[index][0] == feature:
                head, tail = key[:index], key[index + 1 :]
                for centre in (key[index][1] - 1, key[index][1] + 1):
                    if centre:
                        yield head + ((feature, centre),) + tail
                    else:
                        yield head + tail
            else:
                head, tail = key[:index], key[index:]
                yield head + ((feature, -1),) + tail
                yield head + ((feature, 1),) + tail

    def _density(self, grid) -> float:
        return grid.density * self.decaying_factor ** (self._time - grid.last_update)

    def learn_one(self, x, w=1.0):
        if not self._known.issuperset(x):
            self._known.update(x)
            self._features = sorted(self._known)

        if self._margin and self._time % self._gap:
            key = self._dcq_means(x)
        else:
            key = self._grid_of(x)

        grid = self._grids.get(key)
        if grid is None:
            grid = DDStreamGrid(self._time)
            self._grids[key] = grid
        grid.density = grid.density * self.decaying_factor ** (self._time - grid.last_update) + w
        grid.last_update = self._time

        if self._time and self._time % self._gap == 0:
            self._inspect()

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

    def _inspect(self):
        grids = self._grids
        dense = []
        forgotten = []

        now = self._time
        decay = self.decaying_factor
        dense_density = self._dense_density
        sparse_density = self._sparse_density

        changed = False
        for key, grid in grids.items():
            faded = decay ** (now - grid.last_update)
            density = grid.density * faded
            if density >= dense_density:
                attribute = _DENSE
                dense.append(key)
            elif density > sparse_density:
                attribute = _TRANSITIONAL
            else:
                attribute = _SPARSE
                if density < sparse_density * (1 - faded * decay):
                    forgotten.append(key)
            if attribute != grid.attribute:
                grid.attribute = attribute
                changed = True

        for key in forgotten:
            del grids[key]

        self._stale_centers = True
        if not changed:
            return

        for grid in grids.values():
            grid.label = None

        clusters = {}
        label = 0
        for key in dense:
            grid = grids[key]
            if grid.label is not None:
                continue
            grid.label = label
            members = {key}
            queue = deque([key])
            while queue:
                for neighbour in self._neighbours(queue.popleft()):
                    other = grids.get(neighbour)
                    if other is None or other.label is not None or other.attribute == _SPARSE:
                        continue
                    other.label = label
                    members.add(neighbour)
                    queue.append(neighbour)
            clusters[label] = members
            label += 1

        self._clusters = clusters

    def _update_centers(self):
        half = self.grid_width / 2
        centers = {}
        for label, members in self._clusters.items():
            total = 0.0
            center: dict = {}
            for key in members:
                weight = self._density(self._grids[key])
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


class DDStreamGrid:
    """DD-Stream eigenvector."""

    __slots__ = ("density", "last_update", "label", "attribute")

    def __init__(self, last_update):
        self.density = 0.0
        self.last_update = last_update
        self.label: int | None = None
        self.attribute = _SPARSE
