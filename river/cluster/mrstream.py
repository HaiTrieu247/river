from __future__ import annotations

import itertools
import math
import operator
from collections import deque

from river import base

__all__ = ["MRStream"]


class MRStream(base.Clusterer):
    r"""MR-Stream

    MR-Stream [^1] is a density-based clustering algorithm for evolving data streams that
    keeps its synopsis in a tree rather than in a flat grid. The data space is split in half
    along every dimension, then each half is split again, down to `max_height` levels. Every
    tree node is a cell, and the level a cell sits on is a resolution: the same stream can be
    clustered coarsely near the root or finely near the leaves, without touching the online
    phase.

    **Cells and weights**

    A record's weight starts at `1` and fades as $\lambda^{-at}$, written here as a single
    `decaying_factor` $d = \lambda^{-a}$ so that a record is worth $d^{\Delta t}$ after
    $\Delta t$ steps. A cell's weight is the sum of the weights of the records inside it, and
    a cell that receives a record is refreshed in one step:

    $$w(g, t_j) = d^{\,t_j - t_i} w(g, t_i) + 1$$

    Every record is folded into all `max_height` cells that contain it, one per level, so the
    online phase is $O(nH)$ per record and never depends on how much of the stream has gone by.

    **Dense and sparse cells**

    Total weight in the system converges to $\frac{1}{1 - d}$, and the space holds
    `n_cells`$^h$ cells at level $h$, so the average cell weight at that level tends to
    $\frac{1}{n\_cells^h (1 - d)}$. A cell is *dense* when it reaches $C_H$ times that
    average, *sparse* when it falls below $C_L$ times it, and *transitional* in between. Both
    the paper's density form $D(g,t) \ge C_H / V(1 - d)$ and this weight form say the same
    thing, since cell volume cancels once density is written as $w(g,t)/V(g)$.

    **Pruning and merging the tree**

    Three mechanisms keep the tree small. Up-pruning runs after every record: when the cell a
    record landed in is dense and so are all $2^n$ of its siblings, the children are dropped
    and the parent records their total in its *implicit weight*, which stands in for the
    subtree that is no longer stored. Down-pruning runs every `gap` steps and drops leaves
    whose weight has fallen under

    $$p(\Delta t) = \frac{C_L}{n\_cells^h (1 - d)} (1 - d^{\,\Delta t + 1})$$

    The paper derives this shape from three requirements — the threshold must stay under the
    sparse bound, must not remove a cell that could still come back, and must be reachable
    within one inspection — and shows the resulting node count is bounded by
    $v t_p \ln(t_c/t_p)$ for an arrival rate of $v$ records per step. Merging, also every
    `gap` steps, collapses any cell whose $2^n$ children are all leaves of its own class into
    its implicit weight.

    Since $2^n$ children are needed before anything can be merged, both merges are a
    low-dimensional affair; above a handful of features they stop firing and the tree is held
    down by down-pruning alone, which is what the paper expects.

    **Offline clustering**

    Clustering is done on demand at `resolution`, which can be changed at any time with
    `mutate` to look at the same synopsis more or less finely. The tree is walked down to that
    level, stopping early at a leaf or at a cell whose implicit weight says its children are
    not sparse, and the non-sparse cells that walk lands on are grouped by connectivity: two
    cells belong together when the gap between their boxes is under $\epsilon$ finest-level
    intervals, so cells that merely touch are joined for $\epsilon = 1$, and cells with room
    between them for larger $\epsilon$. Only dense cells start a group, as the paper asks for
    the cells reachable from a dense one; a transitional cell joins the group that reaches it
    and is left out when none does. Groups holding fewer than `mu` cells and less than
    `beta` weight are dropped as noise. Because cells at different levels can end up in the
    same group, a cluster may be described coarsely where the stream is uniform and finely
    where it has structure.

    **Memory sampling**

    The paper's other use for the node count is to spot evolving clusters without paying for
    the offline phase: a cluster can only appear or grow by adding nodes, and can only shrink
    by having them fade, so a jump in the node count between two inspections is the signal
    that clusters have moved and it is worth clustering again. `memory_samples` holds that
    series.

    Parameters
    ----------
    max_height
        The height $H$ of the tree. Each dimension ends up cut into $2^H$ intervals. It has a
        lower bound, since the density threshold function is only valid when
        $n\_cells^H \geq C_L (1 - d^{gap}) / (1 - d)$, and the constructor rejects a value
        below it.
    resolution
        The level the offline component clusters at, from `1` to `max_height`. Leaving it to
        `None` means the finest level. This is a mutable attribute: `mutate` it to cluster the
        same tree at another resolution.
    bounds
        The `(lower, upper)` range the data spans, shared by every dimension. Values outside
        it are pulled back to the edge cells.
    decaying_factor
        The weight $d = \lambda^{-a}$ a record keeps from one step to the next, within the
        range `(0, 1)`. The paper's $\lambda = 1.002$, $a = 1$ is `0.998` here.
    dense_threshold
        The $C_H$ coefficient of the dense-cell threshold. Must be greater than `1`.
    sparse_threshold
        The $C_L$ coefficient of the sparse-cell threshold. Must be within the range `(0, 1)`.
    n_cells
        The number of cells one cell is taken to split into, which is what turns the density
        thresholds into weight thresholds. The theoretical value is $2^n$, unusable beyond a
        few dimensions since it makes the thresholds vanish, so it is left to be tuned. The
        default is the two-dimensional value.
    epsilon
        How far apart two cells can be, counted in finest-level intervals, and still be
        neighbours. `1` joins cells that touch.
    mu
        The $\mu$ minimum number of cells in a cluster. A group counts as noise when it is
        under `mu` cells *and* under `beta` weight, so relaxing either one keeps it.
    beta
        The $\beta$ minimum weight of a cluster, counted in records still alive rather than as
        a share of the $\frac{1}{1 - d}$ the stream carries in total. With the defaults, a lone
        cell has to hold two records' worth of live weight to be reported. Noisier streams want
        both of these raised.
    gap
        The $t_p$ inspection interval. When left to `None` it is derived, as the paper's
        smallest number of steps in which a dense cell can decay into a sparse one.

    Attributes
    ----------
    n_clusters
        Number of clusters found at the current resolution.
    clusters
        The cells of each cluster, keyed by cluster label. A cell is a
        `(height, coordinates)` pair, where the coordinates are the `(feature, coordinate)`
        pairs of that cell at that height, those equal to `0` being omitted.
    centers
        The weight-weighted centre of each cluster.
    n_nodes
        Number of cells currently held in the tree.
    memory_samples
        The node count as recorded at each inspection, most recent last.

    References
    ----------
    [^1]: Wan, L., Ng, W. K., Dang, X. H., Yu, P. S. and Zhang, K. (2009). Density-Based
          Clustering of Data Streams at Multiple Resolutions. ACM Transactions on Knowledge
          Discovery from Data, 3(3), Article 14.

    Examples
    --------

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [0.30, 0.30], [0.32, 0.28], [0.29, 0.31], [0.31, 0.32], [0.28, 0.29],
    ...     [0.55, 0.30], [0.57, 0.28], [0.54, 0.31], [0.56, 0.32], [0.53, 0.29],
    ... ]

    >>> mrstream = cluster.MRStream(max_height=3, decaying_factor=0.9)

    >>> for _ in range(10):
    ...     for x, _ in stream.iter_array(X):
    ...         mrstream.learn_one(x)

    >>> mrstream.n_clusters
    2

    >>> mrstream.predict_one({0: 0.30, 1: 0.30})
    0

    >>> mrstream.predict_one({0: 0.55, 1: 0.30})
    1

    One empty cell separates the two groups at the finest level, which is what tells them
    apart. Reading the same tree one level up lands them in cells that touch, and there they
    are a single cluster:

    >>> mrstream.mutate({"resolution": 2})
    >>> mrstream.n_clusters
    1

    >>> sorted(mrstream.clusters[0])
    [(2, ((0, 1), (1, 1))), (2, ((0, 2), (1, 1)))]

    All of it is held in seven cells:

    >>> mrstream.n_nodes
    7

    """

    def __init__(
        self,
        max_height: int = 5,
        resolution: int | None = None,
        bounds: tuple[float, float] = (0.0, 1.0),
        decaying_factor: float = 0.998,
        dense_threshold: float = 3.0,
        sparse_threshold: float = 0.8,
        n_cells: int = 4,
        epsilon: float = 1.0,
        mu: int = 2,
        beta: float = 2.0,
        gap: int | None = None,
    ):
        super().__init__()

        if max_height < 1:
            raise ValueError(
                f"The value of `max_height` (currently {max_height}) must be greater than 0."
            )
        if resolution is not None and not 1 <= resolution <= max_height:
            raise ValueError(
                f"The value of `resolution` (currently {resolution}) must be within the range "
                f"[1, `max_height`] (currently [1, {max_height}])."
            )
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(
                f"The value of `bounds` (currently {bounds}) must be a (lower, upper) pair with "
                "lower smaller than upper."
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
        if n_cells < 2:
            raise ValueError(
                f"The value of `n_cells` (currently {n_cells}) must be greater than 1."
            )
        if epsilon < 0:
            raise ValueError(
                f"The value of `epsilon` (currently {epsilon}) must be greater than or equal to 0."
            )
        if mu < 1:
            raise ValueError(f"The value of `mu` (currently {mu}) must be greater than 0.")
        if beta < 0:
            raise ValueError(
                f"The value of `beta` (currently {beta}) must be greater than or equal to 0."
            )
        if gap is not None and gap < 1:
            raise ValueError(f"The value of `gap` (currently {gap}) must be greater than 0.")

        self.max_height = max_height
        self.resolution = resolution
        self.bounds = bounds
        self.decaying_factor = decaying_factor
        self.dense_threshold = dense_threshold
        self.sparse_threshold = sparse_threshold
        self.n_cells = n_cells
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.gap = gap

        self._gap = gap if gap is not None else self._compute_gap()

        smallest = sparse_threshold * (1 - decaying_factor**self._gap) / (1 - decaying_factor)
        if n_cells**max_height < smallest:
            raise ValueError(
                f"The value of `max_height` (currently {max_height}) must be at least "
                f"{math.ceil(math.log(smallest, n_cells))}, otherwise the density threshold "
                "function cannot both spare a cell that has just been written to and remove one "
                "that has been idle."
            )

        scale = 1 - decaying_factor
        self._dense_weights = [
            dense_threshold / (n_cells**h * scale) for h in range(max_height + 1)
        ]
        self._sparse_weights = [
            sparse_threshold / (n_cells**h * scale) for h in range(max_height + 1)
        ]
        self._span = bounds[1] - bounds[0]
        self._steps = 2**max_height

        self._time = 0
        self._known: set = set()
        self._features: list = []
        self._fanout = 1
        self._root = MRStreamCell(0, (), 0)
        self._n_nodes = 1
        self._inspections = 0
        self._samples: deque = deque(maxlen=100)
        self._clusters: dict[int, set[tuple]] = {}
        self._centers: dict[int, dict] = {}
        self._labels: dict[tuple, int] = {}
        self._anchors: list[tuple[int, list[float]]] = []
        self._cached: tuple | None = None

    @property
    def _mutable_attributes(self):
        return {"resolution"}

    def _compute_gap(self) -> int:
        ratio = self.sparse_threshold / self.dense_threshold
        return max(1, int(math.log(ratio) / math.log(self.decaying_factor)))

    @property
    def _resolution(self) -> int:
        if self.resolution is None:
            return self.max_height
        return min(max(1, self.resolution), self.max_height)

    @property
    def n_clusters(self) -> int:
        self._refresh()
        return len(self._clusters)

    @property
    def clusters(self) -> dict[int, set[tuple]]:
        self._refresh()
        return self._clusters

    @property
    def centers(self) -> dict[int, dict]:
        self._refresh()
        return self._centers

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def memory_samples(self) -> list[int]:
        return list(self._samples)

    @property
    def root(self) -> MRStreamCell:
        return self._root

    def _codes(self, x) -> list[tuple]:
        lower = self.bounds[0]
        span = self._span
        steps = self._steps
        codes = []
        for feature, value in x.items():
            scaled = (value - lower) / span * steps
            if scaled <= 0:
                code = 0
            elif scaled >= steps:
                code = steps - 1
            else:
                code = int(scaled)
            codes.append((feature, code))
        codes.sort()
        return codes

    def learn_one(self, x, w=1.0):
        if not self._known.issuperset(x):
            self._known.update(x)
            self._features = sorted(self._known)
            self._fanout = 1 << len(self._features)

        path = self._update(self._codes(x), w)
        self._prune_up(path)

        if self._time and self._time % self._gap == 0:
            self._inspect()

        self._time += 1

    def _update(self, codes, w) -> list[MRStreamCell]:
        now = self._time
        decay = self.decaying_factor
        height = self.max_height
        depth = len(self._features)

        cell = self._root
        path = [cell]
        carried = 0.0

        for level in range(height + 1):
            cell.weight = cell.weight * decay ** (now - cell.last_update) + carried + w
            cell.last_update = now
            if level == height:
                break

            shift = height - level - 1
            digits = []
            coordinates = []
            for feature, code in codes:
                coordinate = code >> shift
                if coordinate & 1:
                    digits.append(feature)
                if coordinate:
                    coordinates.append((feature, coordinate))

            key = tuple(digits)
            child = cell.children.get(key)
            if child is None:
                child = MRStreamCell(level + 1, tuple(coordinates), now)
                cell.children[key] = child
                self._n_nodes += 1
                carried = math.ldexp(cell.implicit_weight, -depth)
            else:
                carried = 0.0

            cell = child
            path.append(cell)

        return path

    def _prune_up(self, path):
        now = self._time
        decay = self.decaying_factor
        fanout = self._fanout
        dense_weights = self._dense_weights

        level = len(path) - 1
        while level:
            cell = path[level]
            if cell.weight < dense_weights[cell.height]:
                return

            parent = path[level - 1]
            if len(parent.children) != fanout:
                return
            for child in parent.children.values():
                if child.children:
                    return
                faded = child.weight * decay ** (now - child.last_update)
                if faded < dense_weights[child.height]:
                    return

            self._n_nodes -= len(parent.children)
            parent.children.clear()
            parent.implicit_weight = parent.weight
            level -= 1

    def _inspect(self):
        self._samples.append(self._n_nodes)
        self._sweep(self._root)
        self._inspections += 1

    def _sweep(self, cell) -> bool:
        now = self._time
        decay = self.decaying_factor
        children = cell.children

        if children:
            dropped = [key for key, child in children.items() if self._sweep(child)]
            for key in dropped:
                del children[key]
            self._n_nodes -= len(dropped)

        faded = decay ** (now - cell.last_update)
        weight = cell.weight * faded

        if not children:
            return bool(cell.height) and weight < self._sparse_weights[cell.height] * (
                1 - faded * decay
            )

        if len(children) != self._fanout:
            return False

        dense = weight >= self._dense_weights[cell.height]
        if not dense and weight > self._sparse_weights[cell.height]:
            return False

        for child in children.values():
            if child.children:
                return False
            other = child.weight * decay ** (now - child.last_update)
            if dense:
                if other < self._dense_weights[child.height]:
                    return False
            elif other > self._sparse_weights[child.height]:
                return False

        self._n_nodes -= len(children)
        children.clear()
        cell.implicit_weight = weight
        return False

    def predict_one(self, x, w=None):
        self._refresh()
        codes = self._codes(x)
        height = self.max_height
        labels = self._labels

        for level in range(self._resolution + 1):
            shift = height - level
            key = (level, tuple((f, c >> shift) for f, c in codes if c >> shift))
            label = labels.get(key)
            if label is not None:
                return label

        if not self._anchors:
            return 0
        point = [x.get(feature, 0.0) for feature in self._features]
        closest = 0
        shortest = math.inf
        for label, anchor in self._anchors:
            distance = math.dist(point, anchor)
            if distance < shortest:
                shortest = distance
                closest = label
        return closest

    def _refresh(self):
        stamp = (self._inspections, self._resolution)
        if self._cached == stamp or not self._time:
            return
        self._cached = stamp
        self._recluster()

    def _frontier(self) -> list[MRStreamCell]:
        resolution = self._resolution
        sparse_weights = self._sparse_weights
        cells = []
        queue = deque([self._root])
        while queue:
            cell = queue.popleft()
            if (
                cell.height == resolution
                or not cell.children
                or cell.implicit_weight > sparse_weights[cell.height]
            ):
                cells.append(cell)
                continue
            queue.extend(cell.children.values())
        return cells

    def _boxes(self, cells) -> list[list[tuple[int, int]]]:
        height = self.max_height
        features = self._features
        boxes = []
        for cell in cells:
            size = 1 << (height - cell.height)
            coordinates = dict(cell.coordinates)
            boxes.append(
                [
                    (coordinates.get(feature, 0) * size, (coordinates.get(feature, 0) + 1) * size)
                    for feature in features
                ]
            )
        return boxes

    def _adjacency(self, cells, boxes) -> list[list[int]]:
        links: list[list[int]] = [[] for _ in cells]
        reach = self.epsilon * self.epsilon

        by_height: dict[int, list[int]] = {}
        for index, cell in enumerate(cells):
            by_height.setdefault(cell.height, []).append(index)

        heights = sorted(by_height)
        for position, height in enumerate(heights):
            group = by_height[height]
            self._link_within(group, boxes, links, 1 << (self.max_height - height))
            for other in heights[position + 1 :]:
                _link_between(group, by_height[other], boxes, links, reach)

        return links

    def _link_within(self, group, boxes, links, size):
        reach = self.epsilon * self.epsilon
        width = len(self._features)
        radius = math.ceil(self.epsilon / size)
        if not width or not radius or (2 * radius + 1) ** width > 4 * len(group):
            _link_between(group, group, boxes, links, reach)
            return

        corners = {tuple(low for low, _ in boxes[index]): index for index in group}
        for offsets in itertools.product(range(-radius, radius + 1), repeat=width):
            if not any(offsets):
                continue
            shift = [offset * size for offset in offsets]
            for corner, index in corners.items():
                other = corners.get(tuple(map(operator.add, corner, shift)))
                if other is not None and _squared_gap(boxes[index], boxes[other], reach) < reach:
                    links[index].append(other)

    def _recluster(self):
        now = self._time
        decay = self.decaying_factor
        sparse_weights = self._sparse_weights
        dense_weights = self._dense_weights

        cells = []
        weights = []
        seeds = []
        for cell in self._frontier():
            weight = cell.weight * decay ** (now - cell.last_update)
            if weight > sparse_weights[cell.height]:
                cells.append(cell)
                weights.append(weight)
                seeds.append(weight >= dense_weights[cell.height])

        links = self._adjacency(cells, self._boxes(cells))
        groups = []
        unvisited = set(range(len(cells)))
        for seed in range(len(cells)):
            if not seeds[seed] or seed not in unvisited:
                continue
            unvisited.discard(seed)
            group = [seed]
            queue = deque(group)
            while queue:
                for other in links[queue.popleft()]:
                    if other in unvisited:
                        unvisited.discard(other)
                        queue.append(other)
                        group.append(other)
            groups.append(group)

        clusters = {}
        labels = {}
        centers = {}
        label = 0
        lower = self.bounds[0]
        for group in groups:
            total = sum(weights[i] for i in group)
            if len(group) < self.mu and total < self.beta:
                continue
            members = set()
            center: dict = {}
            for i in group:
                cell = cells[i]
                key = (cell.height, cell.coordinates)
                members.add(key)
                labels[key] = label
                coordinates = dict(cell.coordinates)
                width = self._span / (1 << cell.height)
                for feature in self._features:
                    middle = lower + (coordinates.get(feature, 0) + 0.5) * width
                    center[feature] = center.get(feature, 0.0) + weights[i] * middle
            clusters[label] = members
            centers[label] = {feature: value / total for feature, value in center.items()}
            label += 1

        self._clusters = clusters
        self._labels = labels
        self._centers = centers
        self._anchors = [
            (label, [center[feature] for feature in self._features])
            for label, center in centers.items()
        ]


def _link_between(one, other, boxes, links, reach):
    same = one is other
    for position, index in enumerate(one):
        for candidate in other[position + 1 :] if same else other:
            if _squared_gap(boxes[index], boxes[candidate], reach) < reach:
                links[index].append(candidate)
                links[candidate].append(index)


def _squared_gap(one, other, reach) -> float:
    total = 0.0
    for (low, high), (other_low, other_high) in zip(one, other):
        gap = max(low, other_low) - min(high, other_high)
        if gap > 0:
            total += gap * gap
            if total >= reach:
                return total
    return total


class MRStreamCell:
    """MR-Stream tree node."""

    __slots__ = ("height", "coordinates", "weight", "implicit_weight", "last_update", "children")

    def __init__(self, height, coordinates, last_update):
        self.height = height
        self.coordinates = coordinates
        self.weight = 0.0
        self.implicit_weight = 0.0
        self.last_update = last_update
        self.children: dict[tuple, MRStreamCell] = {}
