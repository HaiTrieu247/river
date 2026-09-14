from marks import benchmark
from workloads import binary_stream, high_dim_stream

from river import cluster


@benchmark("cluster")
def test_kmeans_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.KMeans(n_clusters=5, seed=42)
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_kmeans_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.KMeans(n_clusters=5, seed=42)
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_dbstream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.DBSTREAM()
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_sostream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.SOStream()
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_sostream_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.SOStream()
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_birch_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.BIRCH()
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_birch_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.BIRCH()
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_improved_birch_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.ImprovedBIRCH()
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_improved_birch_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.ImprovedBIRCH()
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_repstream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.RepStream()
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_repstream_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.RepStream()
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_dcustream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.DCUStream(n_intervals=10, bounds=(0.0, 10.0))
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_dcustream_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.DCUStream(n_intervals=10, bounds=(0.0, 10.0))
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)


@benchmark("cluster")
def test_hpstream_learn(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]

    def run() -> None:
        model = cluster.HPStream(
            max_clusters=10,
            n_projected_dimensions=2,
            n_samples_init=100,
            spread_radius_factor=6,
            seed=42,
        )
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_hpstream_learn_high_dim(benchmark) -> None:
    # HPStream is meant for wide records, and its cost per record grows with the number of
    # dimensions as well as with the number of clusters, so it gets a 50-feature stream too.
    stream = [x for x, _ in high_dim_stream()][:250]

    def run() -> None:
        model = cluster.HPStream(max_clusters=5, n_samples_init=50, spread_radius_factor=3, seed=42)
        for x in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("cluster")
def test_hpstream_predict(benchmark) -> None:
    stream = [x for x, _ in binary_stream()]
    model = cluster.HPStream(
        max_clusters=10,
        n_projected_dimensions=2,
        n_samples_init=100,
        spread_radius_factor=6,
        seed=42,
    )
    for x in stream:
        model.learn_one(x)

    def run() -> None:
        for x in stream:
            model.predict_one(x)

    benchmark(run)
