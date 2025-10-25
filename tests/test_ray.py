import math

import numpy as np
import pytest

from zapmenot import ray, taichi_accelerator
from zapmenot.acceleration import get_backend

pytestmark = pytest.mark.basic


# test calculation of ray length
# reference: hand calculation
def test_ray_length():
    start = [1, 1, 1]
    end = [2, 2, 2]
    aaa = ray.FiniteLengthRay(start, end)
    assert all(aaa._origin == np.array(start))
    assert aaa.length == pytest.approx(math.sqrt(3.))


# test calculaton of ray properties dir, invdir, sign
# reference: hand calculation
def test_ray_unit_vector():
    start = [1, 1, 1]
    end = [2, 2, 2]
    aaa = ray.FiniteLengthRay(start, end)
    part = 1./math.sqrt(3.)
    # the following creates a vector of numerical tests and then
    # checks to ensure they all came back true
    assert all(aaa._origin == [1, 1, 1])
    assert aaa.length == math.sqrt(3.)
    assert all(aaa.direction == np.array([part, part, part]))
    assert all(aaa.inverse_direction == np.array([1/part, 1/part, 1/part]))
    assert aaa.sign == [0, 0, 0]


# test invalid initializations
def test_ray_error_trapping():
    start = [1, 1, 1, 1]
    end = [2, 2, 2]
    with pytest.raises(ValueError):
        aaa = ray.FiniteLengthRay(start, end)  # start vector too long
    start = [1, 1, 1]
    end = [2, 2, 2, 2]
    with pytest.raises(ValueError):
        aaa = ray.FiniteLengthRay(start, end)  # end vector too long
    start = "start"
    end = [2, 2, 2]
    with pytest.raises(ValueError):
        aaa = ray.FiniteLengthRay(start, end)  # non-numeric start
    start = [1, 1, "start"]
    end = [2, 2, 2]
    with pytest.raises(ValueError):
        aaa = ray.FiniteLengthRay(start, end)  # non-numeric start
    start = [1, 1, 1]
    end = "end"
    with pytest.raises(ValueError):
        aaa = ray.FiniteLengthRay(start, end)  # non-numeric end
    start = [1, 1, 1]
    end = [2, 2, "end"]
    with pytest.raises(ValueError):
        aaa = ray.FiniteLengthRay(start, end)  # non-numeric end


def test_ray_properties():
    start = [1, 1, 1]
    end = [2, 2, 2]
    aaa = ray.FiniteLengthRay(start, end)
    aaa.start = [3, 3, 3]
    assert aaa.start == [3, 3, 3]
    with pytest.raises(ValueError):
        aaa.start = "waldo"
    with pytest.raises(ValueError):
        aaa.start = [3, 3, "waldo"]
    aaa.end = [4, 4, 4]
    assert aaa.end == [4, 4, 4]
    with pytest.raises(ValueError):
        aaa.end = "waldo"
    with pytest.raises(ValueError):
        aaa.end = [4, 4, "waldo"]


def test_build_rays_numpy_backend():
    starts = [[0, 0, 0], [1, 0, 0]]
    end = [0, 0, 2]
    rays = ray.build_rays(starts, end, use_taichi=False)
    assert len(rays) == 2
    assert pytest.approx(rays[0].length) == 2.0
    assert pytest.approx(rays[1].length) == math.sqrt(5)
    assert np.isinf(rays[0].inverse_direction[0])
    assert rays[0].sign == [0, 0, 0]


def test_build_rays_zero_length():
    starts = [[1, 2, 3]]
    end = [1, 2, 3]
    rays = ray.build_rays(starts, end, use_taichi=False)
    assert len(rays) == 1
    assert rays[0].length == 0.0
    assert np.all(rays[0].direction == np.zeros(3))
    assert np.all(np.isinf(rays[0].inverse_direction))
    assert rays[0].sign == [0, 0, 0]


def test_build_rays_taichi_availability():
    starts = [[0, 0, 0]]
    end = [1, 1, 1]
    if taichi_accelerator.is_available():
        rays = ray.build_rays(starts, end, use_taichi=True)
        assert len(rays) == 1
        assert rays[0].length == pytest.approx(math.sqrt(3.0))
    else:
        with pytest.raises(RuntimeError):
            ray.build_rays(starts, end, use_taichi=True)


def test_build_rays_explicit_backend():
    starts = [[0, 0, 0]]
    end = [0, 0, 1]
    backend = get_backend("numpy")
    rays = ray.build_rays(starts, end, backend=backend)
    assert len(rays) == 1
    assert rays[0].length == pytest.approx(1.0)
