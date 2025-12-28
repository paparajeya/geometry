from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Union
import math
from .exceptions import GeometryError

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


Number = Union[int, float]



def _validate_backend(backend: str):
    if backend == "numpy" and np is None:
        raise GeometryError("NumPy backend requested but numpy is not installed.")
    if backend == "torch" and torch is None:
        raise GeometryError("Torch backend requested but torch is not installed.")


@dataclass(slots=True, init=False)
class Point:
    data: Union["np.ndarray", "torch.Tensor"]
    backend: str
    device: str

    def __init__(
        self,
        *args,
        backend: str = "numpy",
        device: str = "cpu",
        dtype: str = "float64",
        **kwargs,
    ):
        # -----------------------------
        # Empty Point
        # -----------------------------
        if not args and not kwargs:
            self.data = None
            self.backend = backend
            self.device = device
            return
    
        _validate_backend(backend)

        coords = self._normalize_input(args, kwargs)

        if len(coords) not in (2, 3):
            raise GeometryError("Point must be 2D or 3D.")

        # -----------------------------
        # Backend materialization
        # -----------------------------
        if backend == "numpy":
            if np is None:
                raise GeometryError("NumPy backend unavailable.")
            self.data = np.asarray(coords, dtype=dtype)
            self.backend = "numpy"
            self.device = "cpu"

        elif backend == "torch":
            if torch is None:
                raise GeometryError("Torch backend unavailable.")
            self.data = torch.tensor(
                coords,
                dtype=getattr(torch, dtype),
                device=device,
            )
            self.backend = "torch"
            self.device = device


    # -------------------------------------------------
    # Input normalization (the core logic)
    # -------------------------------------------------

    @staticmethod
    def _normalize_input(args, kwargs) -> list:
        # Case 1: Keyword arguments
        if kwargs:
            if not all(k in ("x", "y", "z") for k in kwargs):
                raise GeometryError("Only x, y, z allowed as keyword arguments.")
            return [kwargs[k] for k in ("x", "y", "z") if k in kwargs]

        # Case 2: Single positional container
        if len(args) == 1:
            val = args[0]

            # Dict-like
            if isinstance(val, dict):
                if not all(k in val for k in ("x", "y")):
                    raise GeometryError("Dictionary must contain at least x and y.")
                return [val[k] for k in ("x", "y", "z") if k in val]

            # NumPy array
            if np is not None and isinstance(val, np.ndarray):
                return val.tolist()

            # Torch tensor
            if torch is not None and torch.is_tensor(val):
                return val.detach().cpu().tolist()

            # Iterable (tuple, list, etc.)
            if isinstance(val, (list, tuple)):
                return list(val)

            raise GeometryError(f"Unsupported input type: {type(val)}")

        # Case 3: Positional numbers
        if len(args) in (2, 3):
            return list(args)

        raise GeometryError("Invalid Point construction.")


    # --------------------
    # Constructors
    # --------------------

    @classmethod
    def from_coords(
        cls,
        x: Number,
        y: Number,
        z: Optional[Number] = None,
        backend: str = "numpy",
        device: str = "cpu",
        dtype="float64",
    ) -> "Point":
        _validate_backend(backend)

        coords = [x, y] if z is None else [x, y, z]

        if backend == "numpy":
            arr = np.asarray(coords, dtype=dtype)
            return cls(arr, backend="numpy", device="cpu")

        tensor = torch.tensor(coords, dtype=getattr(torch, dtype), device=device)
        return cls(tensor, backend="torch", device=device)

    @classmethod
    def from_iterable(
        cls,
        values: Iterable[Number],
        **kwargs,
    ) -> "Point":
        vals = list(values)
        if len(vals) not in (2, 3):
            raise GeometryError("Point must be 2D or 3D.")
        return cls.from_coords(*vals, **kwargs)


    # --------------------
    # Properties
    # --------------------

    @property
    def dim(self) -> int:
        return int(self.data.shape[0])

    @property
    def is_2d(self) -> bool:
        return self.dim == 2

    @property
    def is_3d(self) -> bool:
        return self.dim == 3
    
    @property
    def x(self):
        if self.data is None:
            return None
        return float(self.data[0])

    @property
    def y(self):
        if self.data is None:
            return None
        return float(self.data[1])

    @property
    def z(self):
        if self.data is None or self.dim < 3:
            return None
        return float(self.data[2])


    # --------------------
    # Representation
    # --------------------

    def __repr__(self) -> str:
        if self.data is None:
            return "Point(empty)"
        vals = ", ".join(f"{v:.4f}" for v in self.to_list())
        return f"Point({vals}, backend={self.backend}, device={self.device})"
    
    
    # --------------------
    # Drawable Protocol
    # --------------------

    def geometry_type(self) -> str:
        return "point"

    def visualise(self, backend="opencv", **style):
        from geometry.visualise.dispatch import visualise
        return visualise(self, backend=backend, **style)


    # --------------------
    # Backend helpers
    # --------------------

    def _ensure_same_dim(self, other: "Point"):
        if self.dim != other.dim:
            raise GeometryError("Dimension mismatch between points.")

    def _new(self, data):
        return Point(data, self.backend, self.device)


    # --------------------
    # Conversions
    # --------------------

    def to_numpy(self):
        if self.backend == "numpy":
            return self.data
        return self.data.detach().cpu().numpy()

    def to_tensor(self, device: Optional[str] = None):
        if torch is None:
            raise GeometryError("Torch not available.")
        if self.backend == "torch":
            return self.data.to(device or self.device)
        return torch.tensor(self.data, device=device or self.device)

    def to_device(self, device: str) -> "Point":
        if self.backend != "torch":
            raise GeometryError("Device transfer only supported for torch backend.")
        return self._new(self.data.to(device))

    def to_list(self):
        if self.backend == "numpy":
            return self.data.tolist()
        return self.data.detach().cpu().tolist()
    

    # --------------------
    # Guard All Geometry Ops Against Empty Points
    # --------------------

    def _ensure_not_empty(self):
        if self.data is None:
            raise GeometryError("Operation not allowed on empty Point.")


    # --------------------
    # Basic Operations
    # --------------------

    def translate(self, *delta: Number) -> "Point":
        self._ensure_not_empty()
        if len(delta) != self.dim:
            raise GeometryError("Translation vector dimension mismatch.")
        return self._new(self.data + self._make(delta))

    def scale(self, factor: Number, origin: Optional["Point"] = None) -> "Point":
        self._ensure_not_empty()
        if origin:
            self._ensure_same_dim(origin)
            return self._new(origin.data + factor * (self.data - origin.data))
        return self._new(self.data * factor)

    def rotate_2d(self, angle_rad: float, origin: Optional["Point"] = None) -> "Point":
        self._ensure_not_empty()
        if not self.is_2d:
            raise GeometryError("2D rotation only supported for 2D points.")

        ox, oy = origin.data if origin else (0, 0)
        x, y = self.data

        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        rx = cos_a * (x - ox) - sin_a * (y - oy) + ox
        ry = sin_a * (x - ox) + cos_a * (y - oy) + oy

        return self._new(self._make([rx, ry]))


    # --------------------
    # Distance & Metrics
    # --------------------

    def distance(self, other: "Point") -> float:
        self._ensure_not_empty()
        other._ensure_not_empty()
        self._ensure_same_dim(other)
        diff = self.data - other.data
        if self.backend == "numpy":
            return float(np.linalg.norm(diff))
        return float(torch.linalg.norm(diff).item())

    def length(self) -> float:
        self._ensure_not_empty()
        if self.backend == "numpy":
            return float(np.linalg.norm(self.data))
        return float(torch.linalg.norm(self.data).item())

    def dot(self, other: "Point") -> float:
        self._ensure_not_empty()
        other._ensure_not_empty()
        self._ensure_same_dim(other)
        if self.backend == "numpy":
            return float(np.dot(self.data, other.data))
        return float(torch.dot(self.data, other.data).item())

    def cross(self, other: "Point") -> "Point":
        self._ensure_not_empty()
        other._ensure_not_empty()
        self._ensure_same_dim(other)
        if not self.is_3d:
            raise GeometryError("Cross product requires 3D points.")
        if self.backend == "numpy":
            return self._new(np.cross(self.data, other.data))
        return self._new(torch.cross(self.data, other.data))

    def midpoint(self, other: "Point") -> "Point":
        self._ensure_not_empty()
        other._ensure_not_empty()
        self._ensure_same_dim(other)
        return self._new((self.data + other.data) / 2)


    # --------------------
    # Operator Overloads
    # --------------------

    def __add__(self, other: Union["Point", Number]):
        self._ensure_not_empty()
        if isinstance(other, Point):
            other._ensure_not_empty()
            self._ensure_same_dim(other)
            return self._new(self.data + other.data)
        return self._new(self.data + other)

    def __sub__(self, other: Union["Point", Number]):
        self._ensure_not_empty()
        if isinstance(other, Point):
            other._ensure_not_empty()
            self._ensure_same_dim(other)
            return self._new(self.data - other.data)
        return self._new(self.data - other)

    def __mul__(self, scalar: Number):
        self._ensure_not_empty()
        return self._new(self.data * scalar)

    def __matmul__(self, other: "Point"):
        self._ensure_not_empty()
        return self.dot(other)


    # --------------------
    # Internal Helpers
    # --------------------

    def _make(self, values):
        self._ensure_not_empty()
        if self.backend == "numpy":
            return np.asarray(values, dtype=self.data.dtype)
        return torch.tensor(values, dtype=self.data.dtype, device=self.device)


    # --------------------
    # Point Validations
    # --------------------

    def almost_equals(self, other: "Point", tol: float = 1e-9) -> bool:
        """
        Tolerance-based equality check.
        """
        self._ensure_not_empty()
        other._ensure_not_empty()
        self._same_dim(other)
        self._same_backend(other)

        if self.backend == "numpy":
            return bool(np.allclose(self.data, other.data, atol=tol))
        return bool(torch.allclose(self.data, other.data, atol=tol))

    def equals(
        self,
        other: "Point",
        *,
        tolerant: bool = False,
        tol: float = 1e-9,
    ) -> bool:
        if self.data is None or other.data is None:
            return self.data is None and other.data is None

        self._same_dim(other)
        self._same_backend(other)

        if tolerant:
            if self.backend == "numpy":
                return bool(np.allclose(self.data, other.data, atol=tol))
            return bool(torch.allclose(self.data, other.data, atol=tol))

        # Exact
        if self.backend == "numpy":
            return bool(np.array_equal(self.data, other.data))
        return bool(torch.equal(self.data, other.data))

    def is_empty(self) -> bool:
        """
        Returns True if point has no data or contains NaN.
        """
        if self.data is None:
            return True

        if self.backend == "numpy":
            return bool(np.isnan(self.data).any())
        return bool(torch.isnan(self.data).any())

    def is_finite(self) -> bool:
        """
        Returns True if all coordinates are finite numbers.
        """
        self._ensure_not_empty()
        if self.backend == "numpy":
            return bool(np.isfinite(self.data).all())
        return bool(torch.isfinite(self.data).all())

    def _same_dim(self, other: "Point"):
        self._ensure_not_empty()
        other._ensure_not_empty()
        if self.dim != other.dim:
            raise GeometryError(
                f"Dimension mismatch: {self.dim}D vs {other.dim}D"
            )

    def _same_backend(self, other: "Point"):
        self._ensure_not_empty()
        other._ensure_not_empty()
        if self.backend != other.backend:
            raise GeometryError(
                f"Backend mismatch: {self.backend} vs {other.backend}"
            )
    

    # --------------------
    # Python Operators (== and !=)
    # --------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.equals(other, tolerant=False)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return not self.__eq__(other)


    # --------------------
    # Conversion Utilities
    # --------------------

    def to_dict(self) -> dict:
        if self.data is None:
            return {}

        d = {"x": self.x, "y": self.y}
        if self.dim == 3:
            d["z"] = self.z
        return d

    def to_list(self) -> list:
        if self.data is None:
            return []

        if self.backend == "numpy":
            return self.data.tolist()
        return self.data.detach().cpu().tolist()

    def to_tuple(self) -> tuple:
        return tuple(self.to_list())

    def to_numpy(self):
        if self.data is None:
            return None

        if self.backend == "numpy":
            return self.data
        return self.data.detach().cpu().numpy()

    def to_tensor(self, device: str | None = None):
        if self.data is None:
            return None

        if torch is None:
            raise GeometryError("Torch is not available.")

        if self.backend == "torch":
            return self.data.to(device or self.device)

        return torch.tensor(
            self.data,
            device=device or "cpu",
            dtype=torch.float64,
        )

    def to_homogeneous(self):
        self._ensure_not_empty()

        if self.backend == "numpy":
            one = np.array([1], dtype=self.data.dtype)
            return np.concatenate([self.data, one])

        one = torch.tensor(
            [1],
            dtype=self.data.dtype,
            device=self.device,
        )
        return torch.cat([self.data, one])

    def to_json(self) -> dict:
        return self.to_dict()

    def astype(self, dtype: str) -> "Point":
        self._ensure_not_empty()

        if self.backend == "numpy":
            return Point(
                self.data.astype(dtype),
                backend="numpy",
            )

        return Point(
            self.data.to(dtype=getattr(torch, dtype)),
            backend="torch",
            device=self.device,
        )

