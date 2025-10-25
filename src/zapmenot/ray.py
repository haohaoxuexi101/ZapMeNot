'''
ZapMeNot - a point kernel photon shielding library
Copyright (C) 2019-2025  C. Alan Ford

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numbers
from collections.abc import Iterable, Sequence

import numpy as np

from .acceleration import BaseAccelerationBackend, get_backend


class FiniteLengthRay:
    """Represents a ray in three-space.

    The FiniteLengthRay object has a defined starting point, a defined end,
    and a resulting direction.

    Parameters
    ----------
    start : :class:`list` or :class:`tuple`
        Defines the starting point of the ray in cartesian coordinates.
    end : :class:`list` or :class:`tuple`
        Defines the ending point of the ray in cartesian coordinates.
    """

    '''
    Attributes
    ----------
    _start
    _end
    _origin : :class:`numpy.ndarray`
        A vector implemenation of the starting point.
    _length : float
        The length of the ray.
    _dir : :class:`numpy.ndarray`
        A numpy vector holding the vector normal of the ray.
    _invdir : :class:`numpy.ndarray`
        A numpy vector holding the inverse of the vector _dir.
    _sign : :class:`numpy.ndarray`
        Indicates the signs of the components of :py:obj:`dir`.
    '''

    def __init__(self, start, end):
        if not FiniteLengthRay._is_validate_vector(start):
            raise ValueError("Invalid ray start")
        if not FiniteLengthRay._is_validate_vector(end):
            raise ValueError("Invalid ray end")
        self._start = start
        self._end = end
        self._regularize()

    @property
    def start(self):
        """:class:`list` : A list defining the starting point of the ray in
        cartesian coordinates."""
        return self._start

    @start.setter
    def start(self, value):
        if not FiniteLengthRay._is_validate_vector(value):
            raise ValueError("Invalid ray start")
        self._start = value
        self._regularize()

    @property
    def end(self):
        """:class:`list` : A list defining the ending point of the ray in
        cartesian coordinates."""
        return self._end

    @end.setter
    def end(self, value):
        if not FiniteLengthRay._is_validate_vector(value):
            raise ValueError("Invalid ray end")
        self._end = value
        self._regularize()

    def _regularize(self):
        """Calculates the mean free path for a given distance and photon energy

        Parameters
        ----------
        energy : float
            The photon energy in MeV
        distance : float
            The distance through the material in cm
        """

        self._origin = np.array(self._start)
        v = np.array(self._end) - self._origin
        self._length = np.linalg.norm(v)
        # direction doesn't matter if the length is zero
        if self._length == 0:
            self._dir = np.zeros(3)
        else:
            self._dir = v / self._length
        with np.errstate(divide='ignore'):
            self._invdir = 1/self._dir  # vector is opposite of vector dir
        self._sign = [0, 0, 0]
        self._sign[0] = int((self._invdir[0] < 0))
        self._sign[1] = int((self._invdir[1] < 0))
        self._sign[2] = int((self._invdir[2] < 0))

    @classmethod
    def from_precomputed(
        cls,
        start,
        end,
        length,
        direction,
        inverse_direction,
        sign,
    ):
        """Construct a :class:`FiniteLengthRay` from pre-computed values."""
        if not cls._is_validate_vector(start):
            raise ValueError("Invalid ray start")
        if not cls._is_validate_vector(end):
            raise ValueError("Invalid ray end")
        direction_arr = np.asarray(direction, dtype=float)
        inverse_direction_arr = np.asarray(inverse_direction, dtype=float)
        sign_list = list(sign)
        if direction_arr.shape != (3,) or inverse_direction_arr.shape != (3,):
            raise ValueError("Ray directions must be length 3")
        if len(sign_list) != 3:
            raise ValueError("Ray sign vector must be length 3")

        obj = cls.__new__(cls)
        obj._start = list(start)
        obj._end = list(end)
        obj._origin = np.array(start, dtype=float)
        obj._length = float(length)
        obj._dir = direction_arr
        obj._invdir = inverse_direction_arr
        obj._sign = [int(sign_list[0]), int(sign_list[1]), int(sign_list[2])]
        return obj

    @property
    def length(self):
        """float: The total length of the ray."""
        return self._length

    @property
    def direction(self):
        """numpy.ndarray: Unit direction vector of the ray."""
        return self._dir

    @property
    def inverse_direction(self):
        """numpy.ndarray: Component-wise inverse of the direction vector."""
        return self._invdir

    @property
    def sign(self):
        """list[int]: Sign flags for the inverse direction vector."""
        return list(self._sign)

    @staticmethod
    def _is_validate_vector(vector):
        # vector should be a list
        if not (isinstance(vector, Iterable)):
            return False
        # vector should have a length of 3
        if not (len(vector) == 3):
            return False
        # each element should be a number
        if not all([isinstance(item, numbers.Number) for item in vector]):
            return False
        return True


def build_rays(
    starts: Sequence[Sequence[float]],
    end: Sequence[float],
    use_taichi: bool = False,
    backend: BaseAccelerationBackend | None = None,
):
    """Construct a batch of :class:`FiniteLengthRay` objects.

    Parameters
    ----------
    starts : sequence of sequence of float
        Start coordinates for each ray.
    end : sequence of float
        Shared end coordinate for the rays.
    use_taichi : bool, optional
        When ``True`` and Taichi is available, use the accelerated backend to
        compute geometry terms.

    Returns
    -------
    list of :class:`FiniteLengthRay`
        Rays corresponding to the provided start points.
    """

    if not isinstance(end, Sequence) or len(end) != 3:
        raise ValueError("Invalid ray end")

    if len(starts) == 0:
        return []

    start_array = np.asarray(starts, dtype=np.float64)
    if start_array.ndim != 2 or start_array.shape[1] != 3:
        raise ValueError("Invalid ray start collection")

    if start_array.size == 0:
        return []

    end_array = np.asarray(end, dtype=np.float64)

    if backend is None:
        if use_taichi:
            try:
                backend = get_backend("taichi")
            except (KeyError, RuntimeError) as exc:
                raise RuntimeError("Taichi acceleration was requested but is unavailable") from exc
        else:
            backend = get_backend()

    if use_taichi and backend.name != "taichi":
        # Explicit user request should be honoured.
        raise RuntimeError("Taichi acceleration was requested but is unavailable")

    lengths, directions, inverse_directions, signs = backend.compute_ray_geometry(
        np.ascontiguousarray(start_array, dtype=np.float64),
        np.ascontiguousarray(end_array, dtype=np.float64),
    )

    end_list = [float(x) for x in end_array.tolist()]
    rays = []
    for idx, start in enumerate(start_array):
        start_list = [float(x) for x in start.tolist()]
        ray_obj = FiniteLengthRay.from_precomputed(
            start_list,
            end_list,
            lengths[idx],
            directions[idx],
            inverse_directions[idx],
            signs[idx],
        )
        rays.append(ray_obj)
    return rays
