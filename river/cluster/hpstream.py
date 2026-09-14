from __future__ import annotations

import math
import random
import warnings
from operator import itemgetter

from river import base
from river.utils.vectordict import euclidean_distance_dict

__all__ = ["HPStream"]

# The offline initialisation alternates between assigning points and recomputing what the
# clusters look like. The paper runs that to convergence; in practice the assignment stops
# moving well before this many rounds, and the cap keeps a pathological sample from stalling
# the first call to `learn_one`.
_INIT_ITERATIONS = 20

# Fraction of the dimensions kept when the user does not say how many. The paper's sensitivity
# study (Figure 13) finds anything between 0.4d and 0.8d gives a good clustering, and its own
# scalability runs use 0.6d.
_DEFAULT_PROJECTION = 0.6

# How many points may be turned away by a cluster with no spread before the model is called
# out. One or two is business as usual right after a cluster is created; a steady stream of
# them means the model has settled into the state the docstring warns about.
_DEGENERATE_TOLERANCE = 32


class HPStream(base.Clusterer):
    r"""HPStream

    HPStream [^1] is a projected clustering algorithm for high-dimensional data streams. In
    high-dimensional space every pair of points is nearly equidistant, so a cluster defined on
    all the dimensions at once says very little. HPStream instead gives each cluster its own
    subset of dimensions — the ones the cluster is tight along — and both the clusters and
    their subsets are revised as the stream moves on.

    **Fading cluster structure**

    A cluster is summarised by the first two moments of the points it holds, each point
    weighted by how stale it is. A point that arrived at $T_i$ is worth
    $f(t) = 2^{-\lambda (t - T_i)}$ at time $t$, so a cluster is described by the $(2d + 1)$
    tuple

    $$\mathcal{FC}(\mathcal{C}, t) = (\overline{FC2^x}, \overline{FC1^x}, W(t))$$

    holding, for every dimension $j$, the weighted sum of squares
    $\sum_k f(t - T_{i_k}) (x^j_{i_k})^2$, the weighted sum $\sum_k f(t - T_{i_k}) x^j_{i_k}$,
    and the total weight $W(t) = \sum_k f(t - T_{i_k})$. Two properties make this cheap to
    keep: the tuples of two clusters add up (Observation 2.1), and a cluster nothing was added
    to since $t$ only needs to be multiplied by $2^{-\lambda \delta t}$ (Observation 2.2). The
    decay is therefore applied lazily, at the moment a point is added. A point can be given a
    weight of its own through the `w` argument of `learn_one`, which is what it counts for in
    the sums above.

    **Projected dimensions**

    Every incoming point is tentatively added to each cluster, and the radius of each cluster
    along each dimension is computed from the tuple above:

    $$r_j^2 = \frac{\overline{FC2^x}_j}{W(t)} - \left(\frac{\overline{FC1^x}_j}{W(t)}\right)^2$$

    The $|\mathcal{FCS}| \cdot l$ smallest of those $|\mathcal{FCS}| \cdot d$ radii win. The
    ranking is global, over every cluster at once, which is why `n_projected_dimensions` is an
    *average*: a tight cluster is allowed to keep more dimensions than a loose one, and a
    cluster the ranking leaves with nothing at all is dropped. Adding the incoming point before
    ranking is what lets a cluster holding very few points pick dimensions at all.

    Section 3 of the paper also offers a radius threshold as an alternative to the rank, which
    `radius_threshold` gives you: a dimension is kept when its radius is below the threshold,
    so the number of projected dimensions is free to change over the course of the stream.
    Since the stream is normalised, the threshold is read in standard deviations.

    **Assignment**

    The point goes to the cluster whose centroid is closest, measured by the Manhattan
    segmental distance — the average absolute difference over that cluster's own dimensions,
    and only those. It is added only if it falls inside the cluster's natural boundary, the
    limiting radius

    $$\tau \sqrt{\frac{\sum_{j \in \mathcal{B}(\mathcal{C})} r_j^2}{d'}}$$

    where $d'$ is the number of dimensions the cluster kept. A point outside every boundary
    starts a cluster of its own; if that takes the model past `max_clusters`, the cluster that
    went longest without being written to is deleted. Noise therefore opens clusters that
    nothing joins, and those are the first to be evicted.

    **Normalisation**

    Radii are compared across dimensions, so the dimensions have to be on the same scale. The
    standard deviation of each dimension is estimated from the initialisation sample — or from
    the first `normalization_interval` points, when there is no offline phase — and every value
    is divided by it. The estimate is refreshed every `normalization_interval` points,
    and the cluster tuples are rescaled along with it — $\overline{FC1^x}_j$ by
    $\sigma_j / \sigma'_j$ and $\overline{FC2^x}_j$ by $\sigma^2_j / \sigma'^2_j$ — so the
    statistics stay in the units the model currently reads.

    **Initialisation**

    The first `n_samples_init` points are buffered rather than clustered. They then feed an
    offline k-means, whose assignment and projected dimensions are refined against each other
    until they settle, and the clusters that come out of it seed the stream. Until that
    happens the model has no clusters and `predict_one` returns `0`. Pass `n_samples_init=0`
    to skip the offline phase, in which case the clusters are grown from the stream alone.

    **Time**

    The clock ticks once every `stream_speed` points, which is how the paper reads a stream:
    its default settings, $\lambda = 0.5$ at 200 points per time unit, halve the weight of a
    point every 400 points. That pair of parameters is what sets how much history a cluster
    holds, so lower `decaying_factor` or raise `stream_speed` to make the model remember
    longer.

    **When the algorithm breaks down**

    Figures 1, 3 and 4 are implemented as written, degenerate cases included, and there is one
    such case you have to keep the model out of. A cluster with no spread — one point, or
    several identical ones — has $r_j = 0$ along every dimension, so Figure 4 gives it a
    limiting radius of $	au \cdot 0 = 0$: no point can ever fall inside it, and it can
    neither grow nor be anything but evicted. Worse, it is *closest* to the points around it,
    so those are turned away too and open clusters of their own, each as dead as the last —
    until every one of the `max_clusters` slots holds a single point and the model is
    predicting noise. Raising $	au$ does not get a model out of that state, because $	au$
    multiplies a radius of zero; it can only keep the model from entering it.

    Such a cluster gets in through one of three doors, and the parameters close all three:

    * The offline k-means hands back a cluster with no spread. Keep `n_samples_init` large
      relative to `max_clusters` — the paper's own `InitNumber` is 2000 — and watch out for
      streams with repeated identical records. A warning is raised if it happens anyway.
    * A point falls outside every boundary and opens a cluster. With `spread_radius_factor` at
      the paper's 2, the boundary sits at twice the radius while a Gaussian point is
      $0.8\sigma$ from its centroid on average, so a percent or so of points fall outside and
      the cascade starts. Raising it to 3 puts the boundary past essentially every point.
    * The ranking of Figure 3 leaves a healthy cluster with no dimension and Figure 1 deletes
      it, which leaves that part of the stream uncovered and its points opening clusters of
      their own. A cluster is safe from this whenever $k \cdot l > (k - 1) \cdot d$, since
      even if the other clusters take every dimension they have there is a slot left over; the
      further apart those two numbers are, the safer. `radius_threshold` sidesteps the problem
      altogether, being read one cluster at a time, and is worth preferring when $d$ is small.

    The paper never met any of the three: at the scale it was evaluated at, no cluster is ever
    created online, so this whole branch of Figure 1 stays dormant. `n_degenerate_clusters`
    reports how many spread-less clusters the model is holding, `n_spawned_clusters` how many
    the online phase has created, and a warning is raised once when the model starts turning
    points away.

    Parameters
    ----------
    max_clusters
        The maximum number $k$ of clusters held at any time. The offline initialisation creates
        this many, and the online phase evicts the least recently written cluster whenever a
        new one takes the model past it.
    n_projected_dimensions
        The average number $l$ of dimensions a cluster is projected on. When left unset, 60% of
        the dimensions seen so far are kept, which is the middle of the range the paper's
        sensitivity study finds safe. Ignored when `radius_threshold` is given.
    radius_threshold
        Keep every dimension whose radius is below this instead of keeping a fixed number of
        them, as suggested at the end of Section 3. Read in standard deviations, since the
        stream is normalised.
    spread_radius_factor
        The factor $\tau$ applied to a cluster's average radius to get the boundary beyond
        which a point starts a cluster of its own.
    decaying_factor
        The rate $\lambda$ at which the weight of a point decays. A point is worth half as much
        after `stream_speed / decaying_factor` further points.
    n_samples_init
        Number of points buffered for the offline initialisation. Use `0` to start clustering
        from the very first point.
    stream_speed
        Number of points that arrive per unit of time.
    normalization_interval
        Number of points between two refreshes of the per-dimension standard deviations.
    seed
        Random seed used to pick the initial centroids of the offline k-means.

    Attributes
    ----------
    n_clusters
        Number of clusters currently held.
    clusters
        The fading cluster structures, keyed by label.
    centers
        The centroid of each cluster, in the units the points arrived in.
    dimensions
        The dimensions each cluster is projected on, keyed by label.

    References
    ----------
    [^1]: Aggarwal, C.C., Han, J., Wang, J. and Yu, P.S. (2004, pp 852-863). A Framework for
          Projected Clustering of High Dimensional Data Streams. In Proceedings of the 30th
          VLDB Conference, Toronto, Canada, 2004.

    Examples
    --------

    >>> from river import cluster

    Three groups of points in three dimensions. The first two dimensions are what tells the
    groups apart; the third is noise, spread over the same range for all of them:

    >>> X = [
    ...     {0: 1.0, 1: 1.0, 2: 0.0}, {0: 1.2, 1: 0.8, 2: 9.0},
    ...     {0: 0.9, 1: 1.1, 2: 4.0}, {0: 1.1, 1: 1.2, 2: 6.0},
    ...     {0: 5.0, 1: 5.0, 2: 8.0}, {0: 5.2, 1: 4.8, 2: 1.0},
    ...     {0: 4.9, 1: 5.1, 2: 5.0}, {0: 5.1, 1: 5.2, 2: 2.0},
    ...     {0: 9.0, 1: 9.0, 2: 3.0}, {0: 9.2, 1: 8.8, 2: 7.0},
    ...     {0: 8.9, 1: 9.1, 2: 0.5}, {0: 9.1, 1: 9.2, 2: 9.5}
    ... ]

    >>> hpstream = cluster.HPStream(
    ...     max_clusters=3,
    ...     n_projected_dimensions=2,
    ...     n_samples_init=12,
    ...     seed=0
    ... )

    >>> for x in X:
    ...     hpstream.learn_one(x)

    >>> hpstream.n_clusters
    3

    Every cluster threw the noise dimension away and kept the two that carry the groups:

    >>> hpstream.dimensions
    {0: [0, 1], 1: [0, 1], 2: [0, 1]}

    >>> hpstream.predict_one({0: 1, 1: 1, 2: 5})
    2

    >>> hpstream.predict_one({0: 5, 1: 5, 2: 5})
    0

    >>> hpstream.predict_one({0: 9, 1: 9, 2: 5})
    1

    What projecting buys you is that clusters living in different subspaces are still found.
    Below, one group is tight along the first two dimensions and noise along the other two,
    and the second group is the other way round — neither is a blob in all four dimensions at
    once, so a full-dimensional clusterer cannot tell them apart:

    >>> import random

    >>> rng = random.Random(42)
    >>> X = []
    >>> for i in range(200):
    ...     if i % 2 == 0:
    ...         X.append({0: rng.gauss(0, 0.1), 1: rng.gauss(0, 0.1),
    ...                   2: rng.uniform(0, 10), 3: rng.uniform(0, 10)})
    ...     else:
    ...         X.append({0: rng.uniform(0, 10), 1: rng.uniform(0, 10),
    ...                   2: rng.gauss(5, 0.1), 3: rng.gauss(5, 0.1)})

    This stream does create clusters online, so `spread_radius_factor` is raised from the
    paper's 2 — see "When the algorithm breaks down" above:

    >>> hpstream = cluster.HPStream(
    ...     max_clusters=2,
    ...     n_projected_dimensions=2,
    ...     spread_radius_factor=3,
    ...     n_samples_init=100,
    ...     seed=0
    ... )

    >>> for x in X:
    ...     hpstream.learn_one(x)

    Each cluster kept the two dimensions it is tight along, listed tightest first:

    >>> hpstream.dimensions
    {0: [3, 2], 1: [0, 1]}

    >>> hpstream.predict_one({0: 0, 1: 0, 2: 3, 3: 8})
    1

    >>> hpstream.predict_one({0: 2, 1: 9, 2: 5, 3: 5})
    0

    No cluster was left holding a single point, which is the state to keep the model out of:

    >>> hpstream.n_degenerate_clusters
    0

    """

    def __init__(
        self,
        max_clusters: int = 10,
        n_projected_dimensions: int | None = None,
        radius_threshold: float | None = None,
        spread_radius_factor: float = 2.0,
        decaying_factor: float = 0.5,
        n_samples_init: int = 2000,
        stream_speed: int = 200,
        normalization_interval: int = 1000,
        seed: int | None = None,
    ):
        super().__init__()

        if max_clusters < 1:
            raise ValueError(
                f"The value of `max_clusters` (currently {max_clusters}) must be at least 1."
            )
        if n_projected_dimensions is not None and n_projected_dimensions < 1:
            raise ValueError(
                f"The value of `n_projected_dimensions` (currently {n_projected_dimensions}) "
                "must be at least 1."
            )
        if radius_threshold is not None and radius_threshold < 0:
            raise ValueError(
                f"The value of `radius_threshold` (currently {radius_threshold}) must not be "
                "negative."
            )
        if spread_radius_factor <= 0:
            raise ValueError(
                f"The value of `spread_radius_factor` (currently {spread_radius_factor}) must "
                "be greater than 0."
            )
        if decaying_factor <= 0:
            raise ValueError(
                f"The value of `decaying_factor` (currently {decaying_factor}) must be greater "
                "than 0."
            )
        if n_samples_init < 0:
            raise ValueError(
                f"The value of `n_samples_init` (currently {n_samples_init}) must not be negative."
            )
        if stream_speed < 1:
            raise ValueError(
                f"The value of `stream_speed` (currently {stream_speed}) must be at least 1."
            )
        if normalization_interval < 1:
            raise ValueError(
                f"The value of `normalization_interval` (currently {normalization_interval}) "
                "must be at least 1."
            )

        self.max_clusters = max_clusters
        self.n_projected_dimensions = n_projected_dimensions
        self.radius_threshold = radius_threshold
        self.spread_radius_factor = spread_radius_factor
        self.decaying_factor = decaying_factor
        self.n_samples_init = n_samples_init
        self.stream_speed = stream_speed
        self.normalization_interval = normalization_interval
        self.seed = seed

        self._decay = 2.0**-decaying_factor
        self._rng = random.Random(seed)

        self._timestamp = 0
        self._n_samples_seen = 0
        self._initialized = n_samples_init == 0
        self._init_buffer: list[tuple[dict, int, float]] = []

        # Bookkeeping for the degenerate state described in the docstring. None of it is read
        # by the clustering itself.
        self._n_spawned = 0
        self._n_turned_away = 0
        self._warned = False

        self._clusters: dict[int, HPStreamFadingCluster] = {}
        # Welford accumulators, as [count, mean, sum of squared deviations] per dimension. The
        # keys double as the register of every dimension the stream has shown so far, which is
        # what the clusters and the projection are read over.
        self._moments: dict[base.typing.FeatureName, list[float]] = {}
        self._sigma: dict[base.typing.FeatureName, float] = {}

    @property
    def n_clusters(self) -> int:
        return len(self._clusters)

    @property
    def clusters(self) -> dict[int, HPStreamFadingCluster]:
        return self._clusters

    @property
    def centers(self) -> dict[int, dict]:
        sigma = self._sigma
        return {
            label: {feature: value * sigma[feature] for feature, value in cluster.center.items()}
            for label, cluster in self._clusters.items()
        }

    @property
    def dimensions(self) -> dict[int, list]:
        return {label: list(cluster.dimensions) for label, cluster in self._clusters.items()}

    @property
    def n_degenerate_clusters(self) -> int:
        """How many clusters have no spread along the dimensions they are projected on.

        Figure 4 gives such a cluster a limiting radius of zero, so it can never take a point.
        A model whose clusters are all degenerate has stopped clustering; see "When the
        algorithm breaks down" in the class docstring.
        """
        return sum(1 for cluster in self._clusters.values() if self._spread(cluster) <= 0)

    @property
    def n_spawned_clusters(self) -> int:
        """How many clusters have been created by the online phase rather than by k-means."""
        return self._n_spawned

    def learn_one(self, x, w=1.0):
        self._n_samples_seen += 1
        # The paper reads the stream at a fixed number of points per unit of time, which is
        # what the decay is expressed in.
        if self._n_samples_seen % self.stream_speed == 0:
            self._timestamp += 1

        self._observe(x)

        if not self._initialized:
            self._init_buffer.append((dict(x), self._timestamp, w))
            if len(self._init_buffer) >= self.n_samples_init:
                self._initialize()
            return

        if self._n_samples_seen % self.normalization_interval == 0:
            self._renormalize()

        # A point worth nothing still says something about the spread of the stream, but it has
        # no business opening a cluster it would then hold with a weight of zero.
        if w > 0:
            self._update(self._normalize(x), w)

    def predict_one(self, x, w=None):
        label, _ = self._closest(self._normalize(x), self._clusters)
        return 0 if label is None else label

    def _observe(self, x):
        """Track the spread of each dimension, and register the ones showing up for the first time."""
        moments = self._moments
        for feature, value in x.items():
            moment = moments.get(feature)
            if moment is None:
                moment = self._register(feature)
            # Welford's online update, as used elsewhere in River through `stats.Var`.
            moment[0] += 1
            delta = value - moment[1]
            moment[1] += delta / moment[0]
            moment[2] += delta * (value - moment[1])

    def _register(self, feature):
        moment = [0.0, 0.0, 0.0]
        self._moments[feature] = moment
        self._sigma[feature] = 1.0
        # Every cluster holds an entry for every known dimension, so that the projection reads
        # the same dimensions for all of them.
        for cluster in self._clusters.values():
            cluster.sum1[feature] = 0.0
            cluster.sum2[feature] = 0.0
        return moment

    def _normalize(self, x):
        sigma = self._sigma
        return {feature: value / sigma.get(feature, 1.0) for feature, value in x.items()}

    def _renormalize(self):
        """Re-estimate the scale of each dimension, and rescale the clusters to match."""
        sigma = self._sigma
        clusters = self._clusters.values()
        for feature, moment in self._moments.items():
            count = moment[0]
            if count < 2:
                continue
            deviation = math.sqrt(moment[2] / (count - 1))
            if deviation <= 0:
                continue
            ratio = sigma[feature] / deviation
            if ratio == 1.0:
                continue
            sigma[feature] = deviation
            for cluster in clusters:
                cluster.sum1[feature] *= ratio
                cluster.sum2[feature] *= ratio * ratio

    def _projection_size(self) -> int:
        d = len(self._sigma)
        if self.n_projected_dimensions is None:
            return max(1, round(_DEFAULT_PROJECTION * d))
        return min(self.n_projected_dimensions, d) if d else self.n_projected_dimensions

    def _compute_dimensions(self, clusters, point, w):
        """Figure 3: rank the radii of every cluster along every dimension, and keep the least."""
        ranked = []
        append = ranked.append
        timestamp = self._timestamp
        decay = self._decay
        get = point.get if point is not None else None

        for label, cluster in clusters.items():
            sum1 = cluster.sum1
            sum2 = cluster.sum2
            if point is None:
                total = cluster.weight
                if total <= 0:
                    continue
                for feature, squares in sum2.items():
                    mean = sum1[feature] / total
                    append((squares / total - mean * mean, label, feature))
            else:
                # The incoming point is added tentatively, which is what makes the radii of a
                # cluster holding very few points mean anything.
                factor = decay ** (timestamp - cluster.last_update)
                total = cluster.weight * factor + w
                if total <= 0:
                    continue
                # `sum1` and `sum2` are written to together and never separately, so they
                # hold the same dimensions in the same order and can be walked side by side.
                for (feature, linear), squares in zip(sum1.items(), sum2.values()):
                    value = get(feature, 0.0)
                    weighted = value * w
                    mean = (linear * factor + weighted) / total
                    append(
                        (
                            (squares * factor + weighted * value) / total - mean * mean,
                            label,
                            feature,
                        )
                    )

        # Ties are broken by the order the clusters were created in, which `sort` keeps, and a
        # cluster ends up holding its dimensions tightest first.
        ranked.sort(key=itemgetter(0))
        projection: dict[int, list] = {label: [] for label in clusters}

        threshold = self.radius_threshold
        if threshold is not None:
            limit = threshold * threshold
            for radius, label, feature in ranked:
                if radius <= limit:
                    projection[label].append(feature)
        else:
            for _, label, feature in ranked[: len(clusters) * self._projection_size()]:
                projection[label].append(feature)

        for label, cluster in clusters.items():
            cluster.dimensions = projection[label]

    def _projected_distance(self, point, cluster) -> float:
        """Figure 2: the Manhattan segmental distance over the dimensions of one cluster."""
        dimensions = cluster.dimensions
        total = cluster.weight
        if not dimensions or total <= 0:
            return math.inf
        sum1 = cluster.sum1
        distance = 0.0
        for feature in dimensions:
            distance += abs(point.get(feature, 0.0) - sum1[feature] / total)
        # Averaging over the dimensions taken part in is what keeps clusters projected on a
        # different number of them comparable.
        return distance / len(dimensions)

    def _closest(self, point, clusters):
        best = None
        best_distance = math.inf
        for label, cluster in clusters.items():
            distance = self._projected_distance(point, cluster)
            if distance < best_distance:
                best_distance = distance
                best = label
        return best, best_distance

    def _spread(self, cluster) -> float:
        """The summed square radius of a cluster over the dimensions it is projected on."""
        total = cluster.weight
        if not cluster.dimensions or total <= 0:
            return 0.0
        sum1 = cluster.sum1
        sum2 = cluster.sum2
        squared = 0.0
        for feature in cluster.dimensions:
            mean = sum1[feature] / total
            squared += sum2[feature] / total - mean * mean
        return squared

    def _limiting_radius(self, cluster) -> float:
        """Figure 4: the natural boundary of a cluster, as a factor of its average radius."""
        squared = self._spread(cluster)
        if squared <= 0:
            # Figure 4 as written. A cluster holding a single point has no spread at all, so
            # its boundary is zero and nothing can ever fall inside it. See "When the algorithm
            # breaks down" in the class docstring: this is the state the parameters have to
            # keep the model out of, and `_check_degenerate` is what tells the user about it.
            return 0.0
        return math.sqrt(squared / len(cluster.dimensions)) * self.spread_radius_factor

    def _update(self, point, w):
        """Figure 1: the steps taken for one point of the stream."""
        if not point:
            # A record with no dimensions has nothing to say about where the clusters are, and
            # a cluster made out of it would be projected on nothing.
            return

        clusters = self._clusters
        self._compute_dimensions(clusters, point, w)

        label, distance = self._closest(point, clusters)
        if label is None:
            boundary = 0.0
        else:
            boundary = self._limiting_radius(clusters[label])
            if boundary <= 0:
                self._n_turned_away += 1
        if label is None or distance > boundary:
            # The point lies outside the natural boundary of every cluster, so it is one of a
            # kind for now. Noise ends up here too, and is evicted below once it goes stale.
            fresh = self._new_cluster(point, w)
            self._n_spawned += 1
        else:
            fresh = None
            self._add_point(clusters[label], point, w)

        for starved in [key for key, cluster in clusters.items() if not cluster.dimensions]:
            del clusters[starved]

        if fresh is not None:
            # Room is made before the newcomer is filed, rather than after as in Figure 1. The
            # newcomer is the most recently written cluster of all, so it is never the one the
            # eviction picks, and doing it in this order lets the label of the evicted cluster
            # be handed straight to it.
            if len(clusters) >= self.max_clusters:
                del clusters[min(clusters, key=lambda key: clusters[key].last_seen)]
            clusters[self._free_label()] = fresh
            self._check_degenerate()

    def _check_degenerate(self):
        """Say something once the model has settled into the state Figure 4 cannot leave."""
        if self._warned or self._n_turned_away < _DEGENERATE_TOLERANCE:
            return
        self._warned = True
        warnings.warn(
            f"HPStream has turned {self._n_turned_away} points away from clusters that hold a "
            "single point. Figure 4 of the paper gives such a cluster a limiting radius of "
            "zero, so it can never grow, and the points it turns away open clusters that are "
            "just as unable to grow. Raise `spread_radius_factor` (3 is usually enough, the "
            "paper's default of 2 assumes a stream that never creates a cluster online), or "
            "raise `n_samples_init` so that the offline k-means starts from clusters that "
            "have some spread.",
            stacklevel=4,
        )

    def _free_label(self) -> int:
        """The smallest label no cluster is holding, so that labels stay within `max_clusters`."""
        clusters = self._clusters
        label = 0
        while label in clusters:
            label += 1
        return label

    def _new_cluster(self, point, w):
        sum1 = dict.fromkeys(self._sigma, 0.0)
        sum2 = dict.fromkeys(self._sigma, 0.0)
        for feature, value in point.items():
            weighted = value * w
            sum1[feature] = weighted
            sum2[feature] = weighted * value
        cluster = HPStreamFadingCluster(sum1, sum2, w, self._timestamp, self._n_samples_seen)
        # A cluster holding a single point has a radius of zero everywhere, so it is born on
        # every dimension of that point and narrowed down by the next ranking.
        cluster.dimensions = list(point)
        return cluster

    def _add_point(self, cluster, point, w):
        sum1 = cluster.sum1
        sum2 = cluster.sum2
        # Observation 2.2, applied lazily: the decay owed since the last write is settled here.
        factor = self._decay ** (self._timestamp - cluster.last_update)
        if factor != 1.0:
            for feature in sum1:
                sum1[feature] *= factor
                sum2[feature] *= factor
            cluster.weight *= factor
        # Observation 2.1: a point is a cluster of its own, so it is simply summed in.
        for feature, value in point.items():
            weighted = value * w
            sum1[feature] += weighted
            sum2[feature] += weighted * value
        cluster.weight += w
        cluster.last_update = self._timestamp
        cluster.last_seen = self._n_samples_seen

    def _initialize(self):
        """Seed the stream with an offline k-means over the points buffered so far."""
        buffer = self._init_buffer
        self._init_buffer = []
        self._initialized = True

        # The buffered sample is what the scale of every dimension is first read from.
        self._renormalize()

        decay = self._decay
        now = self._timestamp
        points = [
            (self._normalize(raw), w * decay ** (now - timestamp))
            for raw, timestamp, w in buffer
            if raw and w > 0
        ]
        if not points:
            return

        k = min(self.max_clusters, len(points))
        clusters = {
            label: self._new_cluster(point, w)
            for label, (point, w) in enumerate(self._seeds(points, k))
        }
        labels = [-1] * len(points)

        # A plain k-means over all the dimensions gives the starting set of clusters.
        for _ in range(_INIT_ITERATIONS):
            if not self._assign(points, clusters, labels, projected=False):
                break
            clusters = self._group(points, labels)

        # The dimensions and the assignment are then refined against each other, since a point
        # is closest to a different cluster once only the projected dimensions are read.
        for _ in range(_INIT_ITERATIONS):
            self._compute_dimensions(clusters, None, 0.0)
            if not self._assign(points, clusters, labels, projected=True):
                break
            clusters = self._group(points, labels)

        self._compute_dimensions(clusters, None, 0.0)
        self._clusters = {
            label: cluster for label, cluster in clusters.items() if cluster.dimensions
        }

        born_dead = self.n_degenerate_clusters
        if born_dead:
            warnings.warn(
                f"The offline initialisation of HPStream produced {born_dead} cluster(s) with "
                "no spread. Figure 4 of the paper gives them a limiting radius of zero, so "
                "they can never take a point and will drag the rest of the model down with "
                "them. Raise `n_samples_init` (currently "
                f"{self.n_samples_init}) or lower `max_clusters` (currently "
                f"{self.max_clusters}) so that every cluster starts from more than one point.",
                stacklevel=3,
            )

    def _seeds(self, points, k):
        """Pick the points the offline k-means starts from, the way k-means++ does.

        The paper leaves the seeding of its k-means open. Spreading the seeds out matters here
        more than it usually does: a projected cluster is separated from the others along a few
        dimensions only, so two seeds landing in the same group leave the k-means in a local
        optimum that splits along a dimension the clusters have nothing to say about, and the
        projection that follows only entrenches it.
        """
        chosen = [points[self._rng.randrange(len(points))]]
        squared = [euclidean_distance_dict(point, chosen[0][0]) ** 2 for point, _ in points]
        while len(chosen) < k:
            total = math.fsum(squared)
            if total <= 0:
                # Every remaining point sits on a seed already, so anything else will do.
                chosen.extend(self._rng.sample(points, k - len(chosen)))
                break
            target = self._rng.random() * total
            index = len(points) - 1
            for candidate, distance in enumerate(squared):
                target -= distance
                if target <= 0:
                    index = candidate
                    break
            chosen.append(points[index])
            center = points[index][0]
            for candidate, (point, _) in enumerate(points):
                distance = euclidean_distance_dict(point, center) ** 2
                if distance < squared[candidate]:
                    squared[candidate] = distance
        return chosen

    def _assign(self, points, clusters, labels, projected):
        """Send each point to its closest cluster, and report whether anything moved."""
        centers = {label: cluster.center for label, cluster in clusters.items()}
        moved = False
        for index, (point, _) in enumerate(points):
            if projected:
                label, _ = self._closest(point, clusters)
            else:
                label = None
                best_distance = math.inf
                for candidate, center in centers.items():
                    distance = euclidean_distance_dict(point, center)
                    if distance < best_distance:
                        best_distance = distance
                        label = candidate
            if label is not None and label != labels[index]:
                labels[index] = label
                moved = True
        return moved

    def _group(self, points, labels):
        clusters: dict[int, HPStreamFadingCluster] = {}
        for (point, w), label in zip(points, labels):
            cluster = clusters.get(label)
            if cluster is None:
                clusters[label] = self._new_cluster(point, w)
            else:
                self._add_point(cluster, point, w)
        # Empty clusters are dropped, and the rest are read in label order so that the ranking
        # of the radii breaks its ties the same way on every round.
        return dict(sorted(clusters.items()))

    @classmethod
    def _unit_test_params(cls):
        yield {"max_clusters": 3, "n_samples_init": 10, "stream_speed": 5, "seed": 42}


class HPStreamFadingCluster:
    """The fading cluster structure of Definition 2.2, with the dimensions it is projected on."""

    __slots__ = ("sum1", "sum2", "weight", "last_update", "last_seen", "dimensions")

    def __init__(self, sum1, sum2, weight, last_update, last_seen):
        self.sum1 = sum1
        self.sum2 = sum2
        self.weight = weight
        self.last_update = last_update
        # The number of points the model had read when this cluster was last written to, which
        # is what the eviction of a stale cluster goes by.
        self.last_seen = last_seen
        self.dimensions: list = []

    @property
    def center(self) -> dict:
        weight = self.weight
        if weight <= 0:
            return dict.fromkeys(self.sum1, 0.0)
        return {feature: value / weight for feature, value in self.sum1.items()}

    def radius(self, feature) -> float:
        """The radius of the cluster along one dimension, as given by Equation 1."""
        weight = self.weight
        if weight <= 0:
            return 0.0
        mean = self.sum1[feature] / weight
        return math.sqrt(max(self.sum2[feature] / weight - mean * mean, 0.0))

    def __repr__(self):
        return (
            f"HPStreamFadingCluster(weight={self.weight:.3f}, "
            f"dimensions={sorted(self.dimensions, key=repr)})"
        )
