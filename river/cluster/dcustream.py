from __future__ import annotations

import math

from river import base

__all__ = ["DCUStream"]


class DCUStream(base.Clusterer):
    r"""DCUStream

    DCUStream [^1] is a grid and density-based clustering algorithm for *uncertain* data
    streams. Every record carries an existence probability $p$ — the chance that the record is
    there at all — which is passed to `learn_one` as the sample weight `w`. Records the stream
    is unsure about count for less in the grid they land in, so a burst of unreliable readings
    does not by itself open a cluster.

    **Grid partition**

    The data space $D = D_1 \times D_2 \times \cdots \times D_d$ is cut into `n_intervals`
    equal portions along every dimension, which makes $K = k_1 k_2 \cdots k_d$ hyper-cubic
    grids, each identified by the intervals $(g_1, g_2, \ldots, g_d)$ it spans. A record is in
    the grid whose interval holds its value in every dimension. Grids no record has landed in
    are not held in memory and take no part in the clustering.

    **Uncertain-tense weight and density (learning)**

    A record $X$ that arrived at $t_s$ with existence probability $p$ is worth

    $$UTW(X, t) = 2^{-\lambda (t - t_s)} p$$

    at time $t$, so its say in the clustering fades as the stream moves on. The
    *uncertain-tense density* of a grid is the sum of the weights of the records inside it,
    $Den(G, t) = \sum_{X \in G} UTW(X, t)$, which Lemma 2 lets us keep incrementally:

    $$Den(G, t_1) = 2^{-\lambda (t_1 - t_0)} Den(G, t_0) + p$$

    Only the grid a record lands in is touched, so the online phase costs the same for every
    record however many grids are being held.

    **Dynamic density threshold**

    Where most density-based algorithms ask the user for a density threshold, DCUStream
    derives one from the stream itself. Lemma 3 bounds the weight of everything seen so far,
    which caps the average grid density; the threshold is set from that bound:

    $$MinPts(t_u, t) = \frac{\sum_{\tau=0}^{t - t_u} 2^{-\lambda \tau}}{2 K (1 - 2^{-\lambda})}$$

    The sum is the weight the whole stream can hold at time $t$, so the threshold starts near
    zero and climbs to its asymptote as the stream fills up. Early on, when no grid has had
    the time to build a density, the bar is low; once the stream is running it settles on the
    value Definition 6 gives. A grid at or above $MinPts$ is *dense*, and *sparse* otherwise.
    Definition 6 states that inequality the other way round, but the paper's own pseudocode
    reads `Den >= MinPts`, which is the sense used here.

    **Clustering**

    Clusters are grown one at a time out of a *core dense-grid* — the densest grid not yet
    spoken for, as in Definition 8 — through the grids that share a face with it
    (Definition 7, at most $2d$ of them). Dense neighbours join the cluster and are searched
    in turn, depth first; sparse neighbours join it as well, since they can hold the boundary
    points of the cluster, but the search stops there. A sparse grid next to nothing dense is
    left out as noise. Because clusters are seeded in order of density, label `0` is always
    the busiest region of the space at the time of the search.

    The search is run on demand: `learn_one` only refreshes the grid it writes to, and the
    clusters are rebuilt on the next call to `predict_one`, `n_clusters`, `clusters` or
    `centers`. Labels are therefore not stable from one search to the next.

    **What $\lambda$ costs you**

    The paper asks for $\lambda \geq 1$, which is enforced here. It is worth knowing what that
    buys: a record loses half its weight at every step, so at $\lambda = 1$ a grid holding one
    record is dense for about $\log_2 K$ steps and sparse after that. The clusters therefore
    describe the recent tail of the stream — a few dozen records at the widths and dimensions
    a stream is usually read at — rather than its whole history. A larger `n_intervals` or a
    wider record buys a longer tail, since both raise $K$; nothing else does.

    Parameters
    ----------
    n_intervals
        The number $k_i$ of equal portions each dimension of the data space is cut into. The
        same count is used for every dimension, so $K = k^d$ where $d$ is the number of
        features seen so far.
    bounds
        The domain region $D_i$ of every dimension, as a `(lower, upper)` pair. Records
        falling outside it are read into the nearest edge interval, since the paper takes the
        data space as given. Data is expected to be normalised to these bounds.
    decaying_factor
        The rate $\lambda$ at which the weight of a record decays, which the paper requires to
        be at least `1`. The experiments in the paper use `3`.
    span
        Number of records to accumulate before the first clustering, which is the `Span` delay
        of the paper's initial clustering process. Until then there are no clusters.

    Attributes
    ----------
    n_clusters
        Number of clusters generated by the algorithm.
    clusters
        The grids of each cluster, keyed by cluster label. A grid is a tuple of
        `(feature, interval)` pairs.
    centers
        The density-weighted centre of each cluster.
    grids
        Every grid currently holding a record.
    min_pts
        The dynamic density threshold at the current time.

    References
    ----------
    [^1]: Yang, Y., Liu, Z., Zhang, J. and Yang, J. (2012, pp 2664-2670). Dynamic
          Density-based Clustering Algorithm over Uncertain Data Streams. In Proceedings of
          the 9th International Conference on Fuzzy Systems and Knowledge Discovery, May
          29-31, 2012, Chongqing, Sichuan, China.

    Examples
    --------

    >>> from river import cluster
    >>> from river import stream

    Two groups of records arriving side by side, in a data space cut into ten intervals per
    dimension:

    >>> X = [
    ...     [1, 0.5], [4, 3.0], [1, 0.75], [4, 3.25], [1, 1.5], [4, 3.5],
    ...     [1, 0.6], [4, 3.1], [1, 1.6], [4, 3.4], [1, 0.7], [4, 3.2]
    ... ]

    >>> dcustream = cluster.DCUStream(n_intervals=10, bounds=(0.0, 10.0))

    >>> for x, _ in stream.iter_array(X):
    ...     dcustream.learn_one(x)

    >>> dcustream.n_clusters
    2

    The busiest group is the one holding label `0`:

    >>> dcustream.predict_one({0: 4, 1: 3})
    0

    >>> dcustream.predict_one({0: 1, 1: 0.5})
    1

    Records the stream is unsure about count for less. The same 50 records fill a grid in both
    models below, but the ones on the right probably are not there at all, and never take the
    grid past the threshold:

    >>> certain = cluster.DCUStream(n_intervals=10)
    >>> unsure = cluster.DCUStream(n_intervals=10)
    >>> for _ in range(50):
    ...     certain.learn_one({0: 0.5}, w=0.95)
    ...     unsure.learn_one({0: 0.5}, w=0.05)

    >>> certain.n_clusters
    1

    >>> unsure.n_clusters
    0

    """

    def __init__(
        self,
        n_intervals: int = 10,
        bounds: tuple = (0.0, 1.0),
        decaying_factor: float = 1.0,
        span: int = 0,
    ):
        super().__init__()

        if n_intervals < 1:
            raise ValueError(
                f"The value of `n_intervals` (currently {n_intervals}) must be at least 1."
            )
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(
                f"The value of `bounds` (currently {bounds}) must be a (lower, upper) pair with "
                "lower < upper."
            )
        if decaying_factor < 1:
            raise ValueError(
                f"The value of `decaying_factor` (currently {decaying_factor}) must be at least 1."
            )
        if span < 0:
            raise ValueError(f"The value of `span` (currently {span}) must not be negative.")

        self.n_intervals = n_intervals
        self.bounds = bounds
        self.decaying_factor = decaying_factor
        self.span = span

        self._decay = 2.0**-decaying_factor
        self._step = (bounds[1] - bounds[0]) / n_intervals
        # Scaling a value up to the interval count is more accurate than dividing it by the
        # interval width, which rounds a value sitting on a boundary into the interval below.
        self._scale = n_intervals / (bounds[1] - bounds[0])

        self._time = 0
        # The threshold ramps up from the moment the model starts reading the stream.
        self._t_u = 0
        self._known: set = set()
        self._grids: dict[tuple, DCUStreamGrid] = {}
        self._clusters: dict[int, set[tuple]] = {}
        self._centers: dict[int, dict] = {}
        self._clustered = False

    @property
    def n_clusters(self) -> int:
        self._recluster()
        return len(self._clusters)

    @property
    def clusters(self) -> dict[int, set[tuple]]:
        self._recluster()
        return self._clusters

    @property
    def centers(self) -> dict[int, dict]:
        self._recluster()
        return self._centers

    @property
    def grids(self) -> dict[tuple, DCUStreamGrid]:
        return self._grids

    @property
    def min_pts(self) -> float:
        return self._min_pts(self._time)

    def _n_grids(self) -> float:
        # K = k_1 k_2 ... k_d over the dimensions the stream has shown so far.
        try:
            return float(self.n_intervals) ** len(self._known)
        except OverflowError:
            return math.inf

    def _min_pts(self, time) -> float:
        # Definition 6, with the geometric sum written in closed form.
        decay = self._decay
        total = (1 - decay ** (time - self._t_u + 1)) / (1 - decay)
        return total / (2 * self._n_grids() * (1 - decay))

    def _grid_of(self, x) -> tuple:
        # Definition 3: the interval each value falls in, along every dimension of the record.
        lower = self.bounds[0]
        scale = self._scale
        last = self.n_intervals - 1
        coordinates = []
        for feature, value in x.items():
            interval = int((value - lower) * scale)
            if interval < 0:
                interval = 0
            elif interval > last:
                interval = last
            coordinates.append((feature, interval))
        coordinates.sort()
        return tuple(coordinates)

    def _neighbours(self, key):
        # Definition 7: the grids sharing a face with this one, which agree on every dimension
        # but one and sit next to it there.
        last = self.n_intervals - 1
        for index, (feature, interval) in enumerate(key):
            head, tail = key[:index], key[index + 1 :]
            if interval > 0:
                yield head + ((feature, interval - 1),) + tail
            if interval < last:
                yield head + ((feature, interval + 1),) + tail

    def _density(self, grid) -> float:
        return grid.density * self._decay ** (self._time - grid.last_update)

    def learn_one(self, x, w=1.0):
        if not self._known.issuperset(x):
            self._known.update(x)

        key = self._grid_of(x)
        grid = self._grids.get(key)
        time = self._time
        if grid is None:
            grid = DCUStreamGrid(time)
            self._grids[key] = grid
        # Lemma 2, which also gives Lemma 1 for a grid that is being opened right now.
        grid.density = grid.density * self._decay ** (time - grid.last_update) + w
        grid.last_update = time

        self._time = time + 1
        self._clustered = False

    def predict_one(self, x, w=None):
        self._recluster()

        grid = self._grids.get(self._grid_of(x))
        if grid is not None and grid.label is not None:
            return grid.label
        return 0

    def _recluster(self):
        if self._clustered:
            return
        self._clustered = True

        if self._time < self.span:
            self._clusters = {}
            self._centers = {}
            return

        time = self._time
        decay = self._decay
        min_pts = self._min_pts(time)

        densities = {}
        dense = []
        for key, grid in self._grids.items():
            density = grid.density * decay ** (time - grid.last_update)
            densities[key] = density
            grid.label = None
            grid.dense = density >= min_pts
            if grid.dense:
                dense.append(key)

        # Definition 8: a cluster is seeded on the densest grid that is still free, which is
        # the core dense-grid of the region it belongs to.
        dense.sort(key=densities.__getitem__, reverse=True)

        clusters = {}
        label = 0
        for key in dense:
            grid = self._grids[key]
            if grid.label is not None:
                continue
            grid.label = label
            members = {key}
            stack = [key]
            while stack:
                for neighbour in self._neighbours(stack.pop()):
                    other = self._grids.get(neighbour)
                    if other is None or other.label is not None:
                        continue
                    other.label = label
                    members.add(neighbour)
                    # A sparse neighbour is kept for its boundary points but is a dead end.
                    if other.dense:
                        stack.append(neighbour)
            clusters[label] = members
            label += 1

        self._clusters = clusters
        self._update_centers(densities)

    def _update_centers(self, densities):
        lower = self.bounds[0]
        step = self._step
        half = step / 2
        centers = {}
        for label, members in self._clusters.items():
            total = 0.0
            center: dict = {}
            for key in members:
                weight = densities[key]
                if weight <= 0:
                    continue
                total += weight
                for feature, interval in key:
                    center[feature] = center.get(feature, 0.0) + weight * (
                        lower + interval * step + half
                    )
            if total:
                centers[label] = {feature: value / total for feature, value in center.items()}
        self._centers = centers


class DCUStreamGrid:
    """DCUStream grid."""

    __slots__ = ("density", "last_update", "label", "dense")

    def __init__(self, last_update):
        self.density = 0.0
        self.last_update = last_update
        self.label: int | None = None
        self.dense = False
