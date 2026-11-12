from config import Config


def test_rng_streams_reproducible_and_independent():
    x = Config(seed=1).rngs()["init"].random()
    y = Config(seed=1).rngs()["init"].random()
    z = Config(seed=1).rngs()["data"].random()
    assert x == y
    assert x != z