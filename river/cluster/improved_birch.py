from __future__ import annotations

import math

from river import base


class ImprovedBIRCH(base.Clusterer):
    def __init__(
        self,
        initial_threshold: float = 0.5,
        modifying_factor: float = 2.0,
        branching_factor: int = 50,
        leaf_capacity: int = 50,
        max_entries: int = 500,
        distance_measure: str = "d2",
        threshold_measure: str = "diameter",
        n_macro_clusters: int | None = None,
        macro_threshold: float | None = None,
        outlier_fraction: float = 0.25,
        outlier_buffer_size: int | None = None,
    ):
        super().__init__()
        self.initial_threshold = initial_threshold
        self.modifying_factor = modifying_factor
        self.branching_factor = branching_factor
        self.leaf_capacity = leaf_capacity
        self.max_entries = max_entries
        self.distance_measure = distance_measure
        self.threshold_measure = threshold_measure
        self.n_macro_clusters = n_macro_clusters
        self.macro_threshold = macro_threshold
        self.outlier_fraction = outlier_fraction
        self.outlier_buffer_size = outlier_buffer_size

        self._measure = _MEASURES[distance_measure]
        self._tightness = _TIGHTNESS[threshold_measure]
        self._budget = max(1, max_entries)
        self._buffer_size = (
            max(1, max_entries // 5) if outlier_buffer_size is None else outlier_buffer_size
        )
        self._initial = float(initial_threshold)
        self._step = max(0.0, (float(modifying_factor) - 1.0) * self._initial)
        self._keys: list = []
        self._positions: dict = {}
        self._root = ImprovedBIRCHNode(True)
        self._n_entries = 0
        self._n_points = 0.0
        self._n_rebuilds = 0
        self._n_threshold_growths = 0
        self._n_neighbor_merges = 0
        self._outliers: list = []
        self._clusters: dict = {}
        self._n_clusters = 0
        self._clustering_is_up_to_date = False

    def _grown(self, threshold: float, steps: int = 1) -> float:
        return threshold + steps * self._step

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
            self._grow(len(keys))
        return [float(x.get(k, 0.0)) for k in keys]

    def _project(self, x):
        return [float(x.get(k, 0.0)) for k in self._keys]

    def _as_dict(self, values):
        return dict(zip(self._keys, values))

    def _grow(self, d):
        stack = [self._root]
        while stack:
            node = stack.pop()
            for feature in node.features:
                feature.grow(d)
            if not node.leaf:
                stack.extend(node.children)
        for feature in self._outliers:
            feature.grow(d)

    def _leaf_features(self):
        features = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.leaf:
                features.extend(node.features)
            else:
                stack.extend(reversed(node.children))
        return features

    def _closest(self, features, entry):
        measure = self._measure
        best = math.inf
        index = None
        for i, feature in enumerate(features):
            value = measure(feature, entry)
            if value < best:
                best = value
                index = i
        return index

    def _seeds(self, features):
        measure = self._measure
        left = 0
        right = 1
        farthest = -math.inf
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                value = measure(features[i], features[j])
                if value > farthest:
                    farthest = value
                    left = i
                    right = j
        return left, right

    def _partition(self, features, children, left, right, capacity):
        measure = self._measure
        seed_left = features[left]
        seed_right = features[right]
        left_features = [seed_left]
        right_features = [seed_right]
        left_children = [children[left]] if children is not None else None
        right_children = [children[right]] if children is not None else None
        for i, feature in enumerate(features):
            if i == left or i == right:
                continue
            if len(right_features) >= capacity:
                target, target_children = left_features, left_children
            elif len(left_features) >= capacity:
                target, target_children = right_features, right_children
            elif measure(feature, seed_left) <= measure(feature, seed_right):
                target, target_children = left_features, left_children
            else:
                target, target_children = right_features, right_children
            target.append(feature)
            if target_children is not None:
                target_children.append(children[i])
        return left_features, left_children, right_features, right_children

    def _split(self, node):
        capacity = self.leaf_capacity if node.leaf else self.branching_factor
        left, right = self._seeds(node.features)
        kept, kept_children, moved, moved_children = self._partition(
            node.features, None if node.leaf else node.children, left, right, capacity
        )
        node.features = kept
        sibling = ImprovedBIRCHNode(node.leaf)
        sibling.features = moved
        if not node.leaf:
            node.children = kept_children
            sibling.children = moved_children
        return sibling, _summary(moved)

    def _refine(self, node, split_index, new_index):
        features = node.features
        if len(features) < 3:
            return
        measure = self._measure
        left = 0
        right = 1
        closest = math.inf
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                value = measure(features[i], features[j])
                if value < closest:
                    closest = value
                    left = i
                    right = j
        if {left, right} == {split_index, new_index}:
            return
        first = node.children[left]
        second = node.children[right]
        capacity = self.leaf_capacity if first.leaf else self.branching_factor
        merged = first.features + second.features
        merged_children = None if first.leaf else first.children + second.children
        if len(merged) <= capacity:
            first.features = merged
            if merged_children is not None:
                first.children = merged_children
            node.features[left] = _summary(merged)
            del node.features[right]
            del node.children[right]
            return
        seed_left, seed_right = self._seeds(merged)
        kept, kept_children, moved, moved_children = self._partition(
            merged, merged_children, seed_left, seed_right, capacity
        )
        first.features = kept
        second.features = moved
        if merged_children is not None:
            first.children = kept_children
            second.children = moved_children
        node.features[left] = _summary(kept)
        node.features[right] = _summary(moved)

    def _absorb_neighbor(self, node, index):
        features = node.features
        if len(features) < 2:
            return
        target = features[index]
        measure = self._measure
        best = math.inf
        neighbor = None
        for i, feature in enumerate(features):
            if i == index:
                continue
            value = measure(target, feature)
            if value < best:
                best = value
                neighbor = i
        if neighbor is None:
            return
        probe = ImprovedBIRCHClusteringFeature.from_point(features[neighbor].center)
        if self._tightness(target, probe) > target.threshold:
            return
        target += features[neighbor]
        del features[neighbor]
        self._n_entries -= 1
        self._n_neighbor_merges += 1

    def _insert(self, node, entry):
        if node.leaf:
            index = self._closest(node.features, entry)
            if index is not None:
                target = node.features[index]
                tightness = self._tightness(target, entry)
                if tightness <= target.threshold:
                    target += entry
                    return None
                grown = self._grown(target.threshold)
                if self._step > 0.0 and tightness <= grown:
                    target += entry
                    target.threshold = grown
                    self._n_threshold_growths += 1
                    self._absorb_neighbor(node, index)
                    return None
            node.features.append(entry)
            self._n_entries += 1
            if len(node.features) <= self.leaf_capacity:
                return None
            return self._split(node)
        index = self._closest(node.features, entry)
        split = self._insert(node.children[index], entry)
        if split is None:
            node.features[index] += entry
            return None
        sibling, sibling_feature = split
        node.features[index] = _summary(node.children[index].features)
        node.features.append(sibling_feature)
        node.children.append(sibling)
        if len(node.features) <= self.branching_factor:
            self._refine(node, index, len(node.features) - 1)
            return None
        return self._split(node)

    def _insert_entry(self, entry):
        split = self._insert(self._root, entry)
        if split is None:
            return
        sibling, sibling_feature = split
        previous = self._root
        root = ImprovedBIRCHNode(False)
        root.features = [_summary(previous.features), sibling_feature]
        root.children = [previous, sibling]
        self._root = root

    def _crowded_leaf(self):
        node = self._root
        while not node.leaf:
            index = 0
            heaviest = -math.inf
            for i, feature in enumerate(node.features):
                if feature.n > heaviest:
                    heaviest = feature.n
                    index = i
            node = node.children[index]
        return node

    def _pair_deficits(self, features):
        tightness = self._tightness
        deficits = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                deficits.append(
                    tightness(features[i], features[j])
                    - max(features[i].threshold, features[j].threshold)
                )
        return deficits

    def _rebuild_steps(self, shed):
        deficits = self._pair_deficits(self._crowded_leaf().features)
        if not deficits:
            deficits = self._pair_deficits(self._leaf_features())
        if not deficits:
            return 1
        deficits.sort()
        return max(1, math.ceil(deficits[min(shed, len(deficits)) - 1] / self._step))

    def _partition_outliers(self, features):
        if self._buffer_size <= 0 or self.outlier_fraction <= 0.0 or len(features) < 2:
            return features, []
        limit = self.outlier_fraction * sum(f.n for f in features) / len(features)
        kept = [f for f in features if f.n >= limit]
        outliers = [f for f in features if f.n < limit]
        if not kept:
            return features, []
        return kept, outliers

    def _absorb_without_growth(self, entry):
        node = self._root
        path = []
        while not node.leaf:
            index = self._closest(node.features, entry)
            if index is None:
                return False
            path.append((node, index))
            node = node.children[index]
        index = self._closest(node.features, entry)
        if index is None:
            return False
        target = node.features[index]
        if self._tightness(target, entry) > target.threshold:
            return False
        target += entry
        for parent, position in path:
            parent.features[position] += entry
        return True

    def _store_outliers(self, outliers):
        if not outliers:
            return
        self._outliers.extend(outliers)
        if len(self._outliers) <= self._buffer_size:
            return
        remaining = [f for f in self._outliers if not self._absorb_without_growth(f)]
        overflow = len(remaining) - self._buffer_size
        self._outliers = remaining[overflow:] if overflow > 0 else remaining

    def _rebuild(self):
        steps = 0
        while self._n_entries > self._budget:
            if self._step <= 0.0:
                return
            before = self._n_entries
            shed = self._n_entries - self._budget + max(1, self._budget // 100)
            steps = max(self._rebuild_steps(shed), 2 * steps)
            kept, outliers = self._partition_outliers(self._leaf_features())
            for feature in kept:
                feature.threshold = self._grown(feature.threshold, steps)
            self._root = ImprovedBIRCHNode(True)
            self._n_entries = 0
            for entry in kept:
                self._insert_entry(entry)
            self._store_outliers(outliers)
            self._n_rebuilds += 1
            if self._n_entries < before:
                steps = 0

    def _agglomerate(self, features):
        measure = self._measure
        target = self.n_macro_clusters
        limit = self.macro_threshold
        active = list(range(len(features)))
        neighbor: dict = {}
        distance: dict = {}

        def refresh(i):
            best = math.inf
            index = -1
            for j in active:
                if j == i:
                    continue
                value = measure(features[i], features[j])
                if value < best:
                    best = value
                    index = j
            neighbor[i] = index
            distance[i] = best

        for i in active:
            refresh(i)
        while len(active) > 1:
            if target is not None and len(active) <= target:
                break
            left = min(active, key=lambda i: (distance[i], i))
            right = neighbor[left]
            if right < 0:
                break
            if left > right:
                left, right = right, left
            if limit is not None and features[left].merged_diameter(features[right]) > limit:
                break
            features[left] += features[right]
            active.remove(right)
            del neighbor[right]
            del distance[right]
            for i in active:
                if i == left or neighbor[i] == left or neighbor[i] == right:
                    refresh(i)
        return [features[i] for i in active]

    def _recluster(self):
        if self._clustering_is_up_to_date:
            return
        features = self._leaf_features()
        if self.n_macro_clusters is not None or self.macro_threshold is not None:
            features = self._agglomerate([feature.copy() for feature in features])
        self._clusters = dict(enumerate(features))
        self._n_clusters = len(self._clusters)
        self._clustering_is_up_to_date = True

    def learn_one(self, x, w=1.0):
        values = self._vectorize(x)
        self._insert_entry(
            ImprovedBIRCHClusteringFeature.from_point(values, w, threshold=self._initial)
        )
        self._n_points += w
        if self._n_entries > self._budget:
            self._rebuild()
        self._clustering_is_up_to_date = False

    def predict_one(self, x, w=None):
        self._recluster()
        if not self._clusters:
            return 0
        query = ImprovedBIRCHClusteringFeature.from_point(self._project(x))
        measure = self._measure
        best = math.inf
        label = 0
        for i, feature in self._clusters.items():
            value = measure(feature, query)
            if value < best:
                best = value
                label = i
        return label

    @property
    def n_clusters(self) -> int:
        self._recluster()
        return self._n_clusters

    @property
    def clusters(self) -> dict[int, ImprovedBIRCHClusteringFeature]:
        self._recluster()
        return self._clusters

    @property
    def centers(self) -> dict:
        self._recluster()
        return {i: self._as_dict(f.center) for i, f in self._clusters.items()}

    @property
    def micro_clusters(self) -> dict[int, ImprovedBIRCHClusteringFeature]:
        return dict(enumerate(self._leaf_features()))

    @property
    def micro_cluster_centers(self) -> dict:
        return {i: self._as_dict(f.center) for i, f in self.micro_clusters.items()}

    @property
    def thresholds(self) -> dict[int, float]:
        return {i: f.threshold for i, f in enumerate(self._leaf_features())}

    @property
    def threshold_step(self) -> float:
        return self._step

    @property
    def n_rebuilds(self) -> int:
        return self._n_rebuilds

    @property
    def n_threshold_growths(self) -> int:
        return self._n_threshold_growths

    @property
    def n_neighbor_merges(self) -> int:
        return self._n_neighbor_merges

    @property
    def outliers(self) -> list[ImprovedBIRCHClusteringFeature]:
        return self._outliers

    @property
    def root(self) -> ImprovedBIRCHNode:
        return self._root

    @property
    def height(self) -> int:
        depth = 1
        node = self._root
        while not node.leaf:
            depth += 1
            node = node.children[0]
        return depth


class ImprovedBIRCHNode:
    __slots__ = ("children", "features", "leaf")

    def __init__(self, leaf: bool):
        self.leaf = leaf
        self.features: list[ImprovedBIRCHClusteringFeature] = []
        self.children: list[ImprovedBIRCHNode] = []


class ImprovedBIRCHClusteringFeature:
    __slots__ = ("linear_sum", "n", "squared_sum", "threshold")

    def __init__(self, n=0.0, linear_sum=None, squared_sum=None, threshold=0.0):
        self.n = float(n)
        self.linear_sum: list[float] = list(linear_sum) if linear_sum is not None else []
        self.squared_sum: list[float] = list(squared_sum) if squared_sum is not None else []
        self.threshold = float(threshold)

    @classmethod
    def from_point(cls, values, weight=1.0, threshold=0.0) -> ImprovedBIRCHClusteringFeature:
        return cls(
            n=weight,
            linear_sum=[weight * v for v in values],
            squared_sum=[weight * v * v for v in values],
            threshold=threshold,
        )

    def copy(self) -> ImprovedBIRCHClusteringFeature:
        return ImprovedBIRCHClusteringFeature(
            self.n, self.linear_sum, self.squared_sum, self.threshold
        )

    def grow(self, d: int) -> None:
        missing = d - len(self.linear_sum)
        if missing > 0:
            self.linear_sum.extend([0.0] * missing)
            self.squared_sum.extend([0.0] * missing)

    def __iadd__(
        self, other: ImprovedBIRCHClusteringFeature
    ) -> ImprovedBIRCHClusteringFeature:
        self.grow(len(other.linear_sum))
        linear_sum = self.linear_sum
        squared_sum = self.squared_sum
        other_squared_sum = other.squared_sum
        for i, value in enumerate(other.linear_sum):
            linear_sum[i] += value
            squared_sum[i] += other_squared_sum[i]
        self.n += other.n
        return self

    def __add__(self, other: ImprovedBIRCHClusteringFeature) -> ImprovedBIRCHClusteringFeature:
        merged = self.copy()
        merged += other
        return merged

    @property
    def weight(self) -> float:
        return self.n

    @property
    def center(self) -> list[float]:
        if self.n <= 0.0:
            return [0.0] * len(self.linear_sum)
        inverse = 1.0 / self.n
        return [value * inverse for value in self.linear_sum]

    def totals(self) -> tuple[float, float]:
        squared_sum = self.squared_sum
        linear = 0.0
        squared = 0.0
        for i, value in enumerate(self.linear_sum):
            linear += value * value
            squared += squared_sum[i]
        return linear, squared

    def radius(self) -> float:
        return _radius(self.n, *self.totals())

    def diameter(self) -> float:
        return _diameter(self.n, *self.totals())

    def _pooled(self, other: ImprovedBIRCHClusteringFeature) -> tuple[float, float, float]:
        linear = 0.0
        squared = 0.0
        other_linear_sum = other.linear_sum
        other_squared_sum = other.squared_sum
        own_linear_sum = self.linear_sum
        own_squared_sum = self.squared_sum
        for i in range(max(len(own_linear_sum), len(other_linear_sum))):
            left = own_linear_sum[i] if i < len(own_linear_sum) else 0.0
            right = other_linear_sum[i] if i < len(other_linear_sum) else 0.0
            total = left + right
            linear += total * total
            squared += (own_squared_sum[i] if i < len(own_squared_sum) else 0.0) + (
                other_squared_sum[i] if i < len(other_squared_sum) else 0.0
            )
        return self.n + other.n, linear, squared

    def merged_radius(self, other: ImprovedBIRCHClusteringFeature) -> float:
        return _radius(*self._pooled(other))

    def merged_diameter(self, other: ImprovedBIRCHClusteringFeature) -> float:
        return _diameter(*self._pooled(other))


def _radius(n: float, linear: float, squared: float) -> float:
    if n <= 0.0:
        return 0.0
    return math.sqrt(max(0.0, squared / n - linear / (n * n)))


def _diameter(n: float, linear: float, squared: float) -> float:
    if n <= 1.0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * (n * squared - linear) / (n * (n - 1.0))))


def _summary(features) -> ImprovedBIRCHClusteringFeature:
    total = ImprovedBIRCHClusteringFeature()
    for feature in features:
        total += feature
    return total


def _centroid_euclidean(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    if a.n <= 0.0 or b.n <= 0.0:
        return math.inf
    left = 1.0 / a.n
    right = 1.0 / b.n
    b_linear_sum = b.linear_sum
    total = 0.0
    for i, value in enumerate(a.linear_sum):
        delta = value * left - b_linear_sum[i] * right
        total += delta * delta
    return math.sqrt(total)


def _centroid_manhattan(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    if a.n <= 0.0 or b.n <= 0.0:
        return math.inf
    left = 1.0 / a.n
    right = 1.0 / b.n
    b_linear_sum = b.linear_sum
    total = 0.0
    for i, value in enumerate(a.linear_sum):
        total += abs(value * left - b_linear_sum[i] * right)
    return total


def _average_inter_cluster(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    if a.n <= 0.0 or b.n <= 0.0:
        return math.inf
    b_linear_sum = b.linear_sum
    b_squared_sum = b.squared_sum
    a_squared_sum = a.squared_sum
    cross = 0.0
    a_squared = 0.0
    b_squared = 0.0
    for i, value in enumerate(a.linear_sum):
        cross += value * b_linear_sum[i]
        a_squared += a_squared_sum[i]
        b_squared += b_squared_sum[i]
    value = (b.n * a_squared + a.n * b_squared - 2.0 * cross) / (a.n * b.n)
    return math.sqrt(max(0.0, value))


def _average_intra_cluster(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    return a.merged_diameter(b)


def _variance_increase(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    if a.n <= 0.0 or b.n <= 0.0:
        return math.inf
    b_linear_sum = b.linear_sum
    total = 0.0
    for i, value in enumerate(a.linear_sum):
        delta = value / a.n - b_linear_sum[i] / b.n
        total += delta * delta
    return math.sqrt(max(0.0, total * a.n * b.n / (a.n + b.n)))


def _merged_radius(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    return a.merged_radius(b)


def _merged_diameter(
    a: ImprovedBIRCHClusteringFeature, b: ImprovedBIRCHClusteringFeature
) -> float:
    return a.merged_diameter(b)


_MEASURES = {
    "d0": _centroid_euclidean,
    "d1": _centroid_manhattan,
    "d2": _average_inter_cluster,
    "d3": _average_intra_cluster,
    "d4": _variance_increase,
}

_TIGHTNESS = {
    "radius": _merged_radius,
    "diameter": _merged_diameter,
}
