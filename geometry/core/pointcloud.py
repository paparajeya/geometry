from __future__ import annotations
from typing import List, Optional, Union
from .exceptions import GeometryError
from .point import Point

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class PointCloud:
    def __init__(
        self,
        data: Optional[Union[
            List['Point'],
            List[List[float]],
            'np.ndarray',
            'torch.Tensor'
        ]] = None,
        backend: str = "auto",
        device: str = "cpu",
        dtype: str = "float64",
    ):
        # Backend selection
        if backend == "auto":
            if TORCH_AVAILABLE:
                backend = "torch"
            elif NUMPY_AVAILABLE:
                backend = "numpy"
            else:
                backend = "plain"

        self.backend = backend
        self.device = device
        self.dtype = dtype

        # Empty initialization
        if data is None or (isinstance(data, list) and len(data) == 0):
            self.data = None
            self.dim = None
            return

        # Normalize input to list of lists or array/tensor
        self.data, self.dim = self._normalize_input(data)

        # Convert to backend
        if self.backend == "numpy":
            if not NUMPY_AVAILABLE:
                raise ImportError("NumPy is not installed")
            self.data = np.asarray(self.data, dtype=self.dtype)
            self.device = "cpu"  # numpy is always CPU
        elif self.backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed")
            torch_dtype = getattr(torch, self.dtype, torch.float64)
            if isinstance(self.data, torch.Tensor):
                self.data = self.data.detach().clone().to(dtype=torch_dtype, device=self.device)
            else:
                self.data = torch.tensor(self.data, dtype=torch_dtype, device=self.device)

        elif self.backend == "plain":
            # keep as Python list of lists
            self.data = [list(p) for p in self.data]
        else:
            raise GeometryError(f"Unsupported backend '{self.backend}'")

    # -----------------------------
    # Internal: Normalize Input
    # -----------------------------
    def _normalize_input(self, data):
        # If list of Points
        if isinstance(data, list) and all(hasattr(p, "to_list") for p in data):
            arr = [p.to_list() for p in data if not p.is_empty()]
        # If list of lists / tuples
        elif isinstance(data, list) and all(isinstance(p, (list, tuple)) for p in data):
            arr = [list(p) for p in data]
        # If numpy array
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            arr = data
        # If torch tensor
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            arr = data
        else:
            raise GeometryError("Unsupported input type for PointCloud")

        # Determine dimension
        if len(arr) == 0:
            dim = None
        else:
            # for torch or numpy, shape[1]
            if hasattr(arr, "shape"):
                dim = arr.shape[1]
            else:
                dim = len(arr[0])

            if dim not in (2, 3):
                raise GeometryError("Points must be 2D or 3D")

            # Check all points same dim
            if hasattr(arr, "__iter__") and not hasattr(arr, "shape"):
                for p in arr:
                    if len(p) != dim:
                        raise GeometryError("All points must have same dimension")

        return arr, dim

    # -----------------------------
    # Basic Properties
    # -----------------------------
    def __repr__(self):
        if self.is_empty():
            return "PointCloud(empty)"
        points_preview = self.to_points()[:5]
        return f"PointCloud({points_preview}{'...' if len(points_preview)<len(self.data) else ''}, total={len(self.data)})"

    @property
    def points(self):
        """Return list of Point objects for the entire cloud"""
        return self.to_points()

    @property
    def shape(self):
        return (0, 0) if self.is_empty() else self.data.shape

    def is_empty(self) -> bool:
        return self.data is None or len(self.data) == 0

    def is_finite(self) -> bool:
        if self.is_empty():
            return True
        if self.backend == "numpy":
            return np.isfinite(self.data).all()
        return torch.isfinite(self.data).all()
    
    def _same_dim(self, other: 'PointCloud'):
        self._ensure_not_empty()
        other._ensure_not_empty()
        if self.dim != other.dim:
            raise GeometryError(
                f"Dimension mismatch: {self.dim}D vs {other.dim}D"
            )

    # -----------------------------
    # Indexing / Slicing
    # -----------------------------
    def __getitem__(self, idx):
        if self.is_empty():
            raise GeometryError("Empty PointCloud")
        if isinstance(idx, int):
            pt_arr = self.data[idx]
            return Point(*pt_arr, backend=self.backend, device=self.device)
        return PointCloud(self.data[idx], backend=self.backend, device=self.device)

    # -----------------------------
    # Vectorized Transformations
    # -----------------------------
    def translate(self, dx, dy, dz=None):
        self._ensure_not_empty()
        vec = [dx, dy] if self.dim == 2 else [dx, dy, dz if dz is not None else 0]
        if self.backend == "numpy":
            new_data = self.data + np.array(vec, dtype=self.data.dtype)
        else:
            new_data = self.data + torch.tensor(vec, dtype=self.data.dtype, device=self.device)
        return PointCloud(new_data, backend=self.backend, device=self.device)

    def scale(self, factor, center=None):
        self._ensure_not_empty()
        if center is None:
            center_vec = np.zeros(self.dim) if self.backend == "numpy" else torch.zeros(self.dim, device=self.device)
        else:
            center_vec = np.array(center.to_list() if isinstance(center, Point) else center, dtype=float)
            if self.backend == "torch":
                center_vec = torch.tensor(center_vec, device=self.device)
        new_data = (self.data - center_vec) * factor + center_vec
        return PointCloud(new_data, backend=self.backend, device=self.device)

    # -----------------------------
    # Utilities
    # -----------------------------
    def centroid(self):
        self._ensure_not_empty()
        if self.backend == "numpy":
            return Point(*self.data.mean(axis=0), backend="numpy")
        return Point(*self.data.mean(dim=0), backend="torch", device=self.device)

    def bounding_box(self):
        self._ensure_not_empty()
        if self.backend == "numpy":
            min_vals = self.data.min(axis=0)
            max_vals = self.data.max(axis=0)
        else:
            min_vals = self.data.min(dim=0).values
            max_vals = self.data.max(dim=0).values
        return min_vals, max_vals

    # -----------------------------
    # Conversions
    # -----------------------------
    def to_numpy(self):
        if self.is_empty():
            return None
        return self.data if self.backend == "numpy" else self.data.detach().cpu().numpy()

    def to_tensor(self, device=None):
        if self.is_empty():
            return None
        if self.backend == "torch":
            return self.data.to(device or self.device)
        return torch.tensor(self.data, device=device or "cpu", dtype=torch.float64)

    def to_list(self):
        if self.is_empty():
            return []
        if self.backend == "numpy":
            return self.data.tolist()
        return self.data.detach().cpu().tolist()

    def to_points(self) -> List['Point']:
        if self.is_empty():
            return []
        # Convert backend data to list of points
        data_list = self.to_list()
        return [Point(*p, backend=self.backend, device=self.device) for p in data_list]


    def to_dicts(self):
        return [p.to_dict() for p in self.to_points()]

    # -----------------------------
    # Internal Checks
    # -----------------------------
    def _ensure_not_empty(self):
        if self.is_empty():
            raise GeometryError("Operation not allowed on empty PointCloud")


    # -----------------------------
    # Vectorized Distance Computations
    # -----------------------------
    
    def distance_to_point(self, point: 'Point', metric: str = "euclidean"):
        self._ensure_not_empty()
        if point.is_empty():
            raise GeometryError("Target point is empty")

        # Convert point to backend-compatible array
        if self.backend == "numpy":
            p_arr = np.array(point.to_list(), dtype=self.data.dtype)
            diff = self.data - p_arr
            if metric == "euclidean":
                return np.linalg.norm(diff, axis=1)
            elif metric == "manhattan":
                return np.abs(diff).sum(axis=1)
            else:
                raise GeometryError(f"Unsupported metric '{metric}'")

        elif self.backend == "torch":
            p_arr = point.to_tensor(device=self.device)
            diff = self.data - p_arr
            if metric == "euclidean":
                return torch.linalg.norm(diff, dim=1)
            elif metric == "manhattan":
                return torch.sum(torch.abs(diff), dim=1)
            else:
                raise GeometryError(f"Unsupported metric '{metric}'")

        else:  # plain Python
            p_list = point.to_list()
            diff = [[a-b for a,b in zip(row, p_list)] for row in self.data]
            if metric == "euclidean":
                return [sum(d_i**2 for d_i in row)**0.5 for row in diff]
            elif metric == "manhattan":
                return [sum(abs(d_i) for d_i in row) for row in diff]


    def distance_to_pointcloud(self, other: 'PointCloud', metric: str = "euclidean"):
        self._ensure_not_empty()
        other._ensure_not_empty()
        self._same_dim(other)

        if self.backend == "numpy":
            diff = self.data[:, np.newaxis, :] - other.data[np.newaxis, :, :]
            if metric == "euclidean":
                return np.linalg.norm(diff, axis=2)
            elif metric == "manhattan":
                return np.abs(diff).sum(axis=2)
            else:
                raise GeometryError(f"Unsupported metric '{metric}'")

        elif self.backend == "torch":
            diff = self.data[:, None, :] - other.data[None, :, :]
            if metric == "euclidean":
                return torch.linalg.norm(diff, dim=2)
            elif metric == "manhattan":
                return torch.sum(torch.abs(diff), dim=2)
            else:
                raise GeometryError(f"Unsupported metric '{metric}'")

        else:  # plain Python
            result = []
            other_list = other.to_list()
            for row in self.data:
                row_result = []
                for col in other_list:
                    diff = [a-b for a,b in zip(row, col)]
                    if metric == "euclidean":
                        row_result.append(sum(d**2 for d in diff)**0.5)
                    elif metric == "manhattan":
                        row_result.append(sum(abs(d) for d in diff))
                result.append(row_result)
            return result

    def nearest_neighbors(self,
                      query: 'PointCloud',
                      k: int = 1,
                      precompute: bool = False):
        """
        Returns (distances, indices) of k nearest neighbors in self for each point in query.
        - precompute=True uses KD-tree if numpy is available
        - Torch backend uses GPU brute-force automatically
        """
        self._ensure_not_empty()
        query._ensure_not_empty()
        self._same_dim(query)

        if self.backend == "numpy":
            try:
                from scipy.spatial import cKDTree
                if precompute:
                    tree = cKDTree(self.data)
                    dist, idx = tree.query(query.data, k=k)
                    return dist, idx
            except ImportError:
                precompute = False

        # fallback: brute-force vectorized
        dist_matrix = query.distance_to_pointcloud(self)
        
        if self.backend == "torch":
            # dist_matrix is NxM tensor
            if k == 1:
                min_dist, min_idx = torch.min(dist_matrix, dim=1)
            else:
                min_dist, min_idx = torch.topk(dist_matrix, k, largest=False)
            return min_dist, min_idx
        elif self.backend == "numpy":
            if k == 1:
                min_idx = dist_matrix.argmin(axis=1)
                min_dist = dist_matrix[np.arange(dist_matrix.shape[0]), min_idx]
            else:
                min_idx = np.argsort(dist_matrix, axis=1)[:, :k]
                min_dist = np.take_along_axis(dist_matrix, min_idx, axis=1)
            return min_dist, min_idx
        else:  # plain Python
            min_dist, min_idx = [], []
            for row in dist_matrix:
                if k == 1:
                    min_val = min(row)
                    min_dist.append(min_val)
                    min_idx.append(row.index(min_val))
                else:
                    idxs = sorted(range(len(row)), key=lambda i: row[i])[:k]
                    min_idx.append(idxs)
                    min_dist.append([row[i] for i in idxs])
            return min_dist, min_idx



    # -----------------------------
    # Affine Transformations
    # -----------------------------
    def batch_transform(self, matrix=None, translate=None):
        """
        Apply affine transformation to all points in the cloud:
        - matrix: 2x2 or 3x3 rotation/scale/shear
        - translate: [dx, dy] or [dx, dy, dz]
        Returns new PointCloud
        """
        self._ensure_not_empty()

        # Backend conversion helpers
        def to_backend_array(arr):
            if self.backend == "numpy":
                return np.array(arr, dtype=self.data.dtype)
            elif self.backend == "torch":
                return torch.tensor(arr, dtype=self.data.dtype, device=self.device)
            else:
                return arr  # plain Python list

        # Apply matrix
        if matrix is not None:
            if self.backend in ["numpy", "torch"]:
                new_data = self.data @ to_backend_array(matrix).T
            else:  # plain Python
                new_data = [[sum(a*b for a,b in zip(row, col)) for col in zip(*matrix)] for row in self.data]
        else:
            new_data = self.data

        # Apply translation
        if translate is not None:
            translate_vec = to_backend_array(translate)
            if self.backend in ["numpy", "torch"]:
                new_data = new_data + translate_vec
            else:
                new_data = [[x+y for x,y in zip(row, translate_vec)] for row in new_data]

        return PointCloud(new_data, backend=self.backend, device=self.device)


    def add_point(self, point: 'Point', index: Optional[int] = None):
        """Add a new point at the given index (default at end)"""
        new_point = point.to_list()

        if self.is_empty():
            self.data = [new_point] if self.backend == "plain" else None  # handled below
            self.dim = len(new_point)
        else:
            if self.backend == "numpy":
                self.data = np.insert(self.data, index if index is not None else self.data.shape[0], new_point, axis=0)
            elif self.backend == "torch":
                new_tensor = point.to_tensor(device=self.device).unsqueeze(0)
                if index is None or index >= self.data.shape[0]:
                    self.data = torch.cat([self.data, new_tensor], dim=0)
                else:
                    self.data = torch.cat([self.data[:index], new_tensor, self.data[index:]], dim=0)
            else:
                if index is None or index >= len(self.data):
                    self.data.append(new_point)
                else:
                    self.data.insert(index, new_point)

        self._recalculate_bounds()
    
    def add_points(
        self,
        points: Union[
            List[Point],
            List[List[float]],
            List[tuple],
            np.ndarray,
            "torch.Tensor",
            "PointCloud"
        ],
        index: Optional[int] = None
    ):
        """
        Add multiple points to the point cloud.

        Supports:
        - List[Point]
        - List[List[float]] / List[tuple]
        - numpy.ndarray (NxD)
        - torch.Tensor (NxD)
        - PointCloud

        index:
            If provided, inserts at index; otherwise appends.
        """
        if points is None:
            return

        # -------------------------
        # Normalize input
        # -------------------------
        if isinstance(points, PointCloud):
            new_data = points.data
            new_dim = points.dim

        elif isinstance(points, list) and len(points) > 0 and isinstance(points[0], Point):
            new_data = [p.to_list() for p in points]
            new_dim = len(new_data[0])

        elif isinstance(points, (list, tuple)) and len(points) > 0:
            new_data = [list(p) for p in points]
            new_dim = len(new_data[0])

        elif isinstance(points, np.ndarray):
            if points.ndim != 2:
                raise GeometryError("NumPy array must be 2D (NxD)")
            new_data = points
            new_dim = points.shape[1]

        elif TORCH_AVAILABLE and isinstance(points, torch.Tensor):
            if points.ndim != 2:
                raise GeometryError("Torch tensor must be 2D (NxD)")
            new_data = points
            new_dim = points.shape[1]

        else:
            raise GeometryError("Unsupported input type for add_points")

        # -------------------------
        # Initialize empty cloud
        # -------------------------
        if self.is_empty():
            self.dim = new_dim
            if self.backend == "numpy":
                self.data = np.asarray(new_data, dtype=self.dtype)
            elif self.backend == "torch":
                torch_dtype = getattr(torch, self.dtype, torch.float64)
                self.data = (
                    new_data.detach().clone().to(dtype=torch_dtype, device=self.device)
                    if isinstance(new_data, torch.Tensor)
                    else torch.tensor(new_data, dtype=torch_dtype, device=self.device)
                )
            else:
                self.data = new_data.tolist() if not isinstance(new_data, list) else new_data

            self._recalculate_bounds()
            return

        # -------------------------
        # Validation
        # -------------------------
        if new_dim != self.dim:
            raise GeometryError(
                f"Dimension mismatch: PointCloud is {self.dim}D, new points are {new_dim}D"
            )

        # -------------------------
        # Backend-specific insertion
        # -------------------------
        if self.backend == "numpy":
            insert_at = index if index is not None else self.data.shape[0]
            self.data = np.insert(self.data, insert_at, new_data, axis=0)

        elif self.backend == "torch":
            if not isinstance(new_data, torch.Tensor):
                torch_dtype = getattr(torch, self.dtype, torch.float64)
                new_data = torch.tensor(
                    new_data, dtype=torch_dtype, device=self.device
                )
            else:
                new_data = new_data.detach().clone().to(
                    dtype=self.data.dtype, device=self.device
                )

            if index is None or index >= self.data.shape[0]:
                self.data = torch.cat([self.data, new_data], dim=0)
            else:
                self.data = torch.cat(
                    [self.data[:index], new_data, self.data[index:]], dim=0
                )

        else:  # plain Python
            insert_at = index if index is not None else len(self.data)
            self.data[insert_at:insert_at] = (
                new_data if isinstance(new_data, list) else new_data.tolist()
            )

        # -------------------------
        # Update bounds once
        # -------------------------
        self._recalculate_bounds()

        
    def slice(self, start: int = 0, end: Optional[int] = None) -> 'PointCloud':
        """Return a new PointCloud with points between start and end"""
        if self.is_empty():
            return PointCloud(backend=self.backend, device=self.device)

        if self.backend == "numpy":
            sliced = self.data[start:end]
        elif self.backend == "torch":
            sliced = self.data[start:end]
        else:
            sliced = self.data[start:end]

        return PointCloud(sliced, backend=self.backend, device=self.device)

    def add_pointcloud(self, pc: 'PointCloud', index: Optional[int] = None):
        """Append another PointCloud at a given index"""
        if pc.is_empty():
            return

        if self.is_empty():
            self.data = pc.data if self.backend != "torch" else pc.data.clone()
            self.dim = pc.dim
            self._recalculate_bounds()
            return

        if self.backend != pc.backend:
            raise GeometryError("Backends do not match")

        if self.backend == "numpy":
            idx = index if index is not None else self.data.shape[0]
            self.data = np.insert(self.data, idx, pc.data, axis=0)
        elif self.backend == "torch":
            if index is None or index >= self.data.shape[0]:
                self.data = torch.cat([self.data, pc.data], dim=0)
            else:
                self.data = torch.cat([self.data[:index], pc.data, self.data[index:]], dim=0)
        else:
            idx = index if index is not None else len(self.data)
            self.data[idx:idx] = pc.data

        self._recalculate_bounds()

    def arrange(self, algorithm: Optional[str] = "sort_x", custom_func=None):
        """
        Rearrange points:
        - algorithm: "sort_x", "sort_y", "sort_z", "distance_from_origin"
        - custom_func: a callable that takes list/tensor/array and returns rearranged one
        """
        if self.is_empty():
            return

        if custom_func:
            self.data = custom_func(self.data)
            self._recalculate_bounds()
            return

        if self.backend == "numpy":
            if algorithm == "sort_x":
                self.data = self.data[np.argsort(self.data[:,0])]
            elif algorithm == "sort_y":
                self.data = self.data[np.argsort(self.data[:,1])]
            elif algorithm == "sort_z" and self.dim==3:
                self.data = self.data[np.argsort(self.data[:,2])]
            elif algorithm == "distance_from_origin":
                dist = np.linalg.norm(self.data, axis=1)
                self.data = self.data[np.argsort(dist)]
        elif self.backend == "torch":
            if algorithm == "sort_x":
                self.data = self.data[self.data[:,0].argsort()]
            elif algorithm == "sort_y":
                self.data = self.data[self.data[:,1].argsort()]
            elif algorithm == "sort_z" and self.dim==3:
                self.data = self.data[self.data[:,2].argsort()]
            elif algorithm == "distance_from_origin":
                dist = torch.linalg.norm(self.data, dim=1)
                self.data = self.data[dist.argsort()]
        else:
            if algorithm=="sort_x":
                self.data.sort(key=lambda p: p[0])
            elif algorithm=="sort_y":
                self.data.sort(key=lambda p: p[1])
            elif algorithm=="sort_z" and self.dim==3:
                self.data.sort(key=lambda p: p[2])
            elif algorithm=="distance_from_origin":
                self.data.sort(key=lambda p: sum(c**2 for c in p)**0.5)

        self._recalculate_bounds()

    def bounding_box(self):
        """Return (min_xyz, max_xyz) as tuple"""
        if self.is_empty():
            return None, None
        return (self._mins.copy(), self._maxs.copy())

    def _recalculate_bounds(self):
        if self.is_empty():
            self._mins = None
            self._maxs = None
            return

        if self.backend == "numpy":
            self._mins = self.data.min(axis=0).tolist()
            self._maxs = self.data.max(axis=0).tolist()
        elif self.backend == "torch":
            self._mins = self.data.min(dim=0).values.tolist()
            self._maxs = self.data.max(dim=0).values.tolist()
        else:
            transposed = list(zip(*self.data))
            self._mins = [min(coords) for coords in transposed]
            self._maxs = [max(coords) for coords in transposed]
            
    def clear(self):
        """Remove all points from the PointCloud"""
        self.data = None
        self.dim = None
        self._mins = None
        self._maxs = None
        
    def to_2d(self):
        """Return a new 2D PointCloud by dropping z"""
        if self.is_empty() or self.dim==2:
            return self.slice()
        
        if self.backend == "numpy":
            return PointCloud(self.data[:,:2], backend="numpy")
        elif self.backend == "torch":
            return PointCloud(self.data[:,:2], backend="torch", device=self.device)
        else:
            return PointCloud([p[:2] for p in self.data], backend="plain")

    def points_between(self, p1: 'Point', p2: 'Point', tol: float):
        """Return points lying within tol distance to the segment p1-p2"""
        if self.is_empty():
            return PointCloud(backend=self.backend, device=self.device)

        a = p1.to_numpy() if self.backend=="numpy" else p1.to_tensor(device=self.device)
        b = p2.to_numpy() if self.backend=="numpy" else p2.to_tensor(device=self.device)
        
        if self.backend=="numpy":
            pa = self.data - a
            ba = b - a
            t = np.clip(np.sum(pa*ba, axis=1)/np.sum(ba*ba), 0, 1)
            proj = a + (t[:,None]*ba)
            dist = np.linalg.norm(self.data - proj, axis=1)
            return PointCloud(self.data[dist<=tol], backend="numpy")
        elif self.backend=="torch":
            pa = self.data - a
            ba = b - a
            t = torch.clamp((pa*ba).sum(dim=1)/(ba*ba).sum(), 0, 1)
            proj = a + t[:,None]*ba
            dist = torch.linalg.norm(self.data - proj, dim=1)
            return PointCloud(self.data[dist<=tol], backend="torch", device=self.device)
        else:
            # plain Python (loop)
            res = []
            a_list, b_list = p1.to_list(), p2.to_list()
            for pt in self.data:
                pa = [pt[i]-a_list[i] for i in range(self.dim)]
                ba = [b_list[i]-a_list[i] for i in range(self.dim)]
                dot = sum(pa[i]*ba[i] for i in range(self.dim))
                norm2 = sum(c**2 for c in ba)
                t = max(0, min(1, dot/norm2 if norm2>0 else 0))
                proj = [a_list[i]+t*ba[i] for i in range(self.dim)]
                dist = sum((pt[i]-proj[i])**2 for i in range(self.dim))**0.5
                if dist<=tol:
                    res.append(pt)
            return PointCloud(res, backend="plain")

    def points_at_y(self, y_val: float, tol: float = 1e-6):
        if self.is_empty():
            return PointCloud(backend=self.backend, device=self.device)
        
        if self.backend=="numpy":
            mask = np.abs(self.data[:,1] - y_val) <= tol
            return PointCloud(self.data[mask], backend="numpy")
        elif self.backend=="torch":
            mask = torch.abs(self.data[:,1]-y_val) <= tol
            return PointCloud(self.data[mask], backend="torch", device=self.device)
        else:
            return PointCloud([p for p in self.data if abs(p[1]-y_val)<=tol], backend="plain")

    def points_at_x(self, x_val: float, tol: float = 1e-6):
        if self.is_empty():
            return PointCloud(backend=self.backend, device=self.device)
        
        if self.backend=="numpy":
            mask = np.abs(self.data[:,0]-x_val) <= tol
            return PointCloud(self.data[mask], backend="numpy")
        elif self.backend=="torch":
            mask = torch.abs(self.data[:,0]-x_val) <= tol
            return PointCloud(self.data[mask], backend="torch", device=self.device)
        else:
            return PointCloud([p for p in self.data if abs(p[0]-x_val)<=tol], backend="plain")

    def points_at_z(self, z_val: float, tol: float = 1e-6):
        if self.is_empty() or self.dim<3:
            return PointCloud(backend=self.backend, device=self.device)
        
        if self.backend=="numpy":
            mask = np.abs(self.data[:,2]-z_val) <= tol
            return PointCloud(self.data[mask], backend="numpy")
        elif self.backend=="torch":
            mask = torch.abs(self.data[:,2]-z_val) <= tol
            return PointCloud(self.data[mask], backend="torch", device=self.device)
        else:
            return PointCloud([p for p in self.data if abs(p[2]-z_val)<=tol], backend="plain")

    def flip(self, axes: list):
        """
        Flip point cloud across axes
        axes: list of 'x', 'y', 'z'
        """
        if self.is_empty():
            return
        axes_map = {"x":0, "y":1, "z":2}
        for ax in axes:
            idx = axes_map[ax]
            if self.backend=="numpy":
                self.data[:,idx] *= -1
            elif self.backend=="torch":
                self.data[:,idx] *= -1
            else:
                for p in self.data:
                    p[idx] *= -1
        self._recalculate_bounds()
        
    def copy(self):
        """Return a deep copy of the PointCloud"""
        if self.is_empty():
            return PointCloud(backend=self.backend, device=self.device)
        if self.backend=="numpy":
            return PointCloud(self.data.copy(), backend="numpy")
        elif self.backend=="torch":
            return PointCloud(self.data.clone(), backend="torch", device=self.device)
        else:
            return PointCloud([p.copy() for p in self.data], backend="plain")

    def nearest_point(self, point: 'Point', radius: Optional[float] = None):
        """
        Find the nearest point in the cloud to the given point.
        Returns: (Point, index, distance)
        If radius is provided, only consider points within radius.
        """
        if self.is_empty():
            return None, None, None

        # Compute distances
        if self.backend=="numpy":
            p_arr = np.array(point.to_list(), dtype=self.data.dtype)
            dists = np.linalg.norm(self.data - p_arr, axis=1)
        elif self.backend=="torch":
            p_arr = point.to_tensor(device=self.device)
            dists = torch.linalg.norm(self.data - p_arr, dim=1)
        else:
            p_list = point.to_list()
            dists = [sum((c - p_list[i])**2 for i, c in enumerate(pt))**0.5 for pt in self.data]

        # Apply radius filter
        if radius is not None:
            if self.backend in ["numpy","torch"]:
                mask = dists <= radius
                dists_masked = dists[mask]
                idxs = np.where(mask)[0] if self.backend=="numpy" else torch.nonzero(mask).flatten()
            else:
                idxs = [i for i,d in enumerate(dists) if d<=radius]
                dists_masked = [dists[i] for i in idxs]

            if len(idxs)==0:
                return None, None, None
            if self.backend=="torch":
                min_val, min_idx = torch.min(dists_masked, dim=0)
                min_idx = idxs[min_idx.item()]
            elif self.backend=="numpy":
                min_idx = idxs[np.argmin(dists_masked)]
                min_val = dists_masked[np.argmin(dists_masked)]
            else:
                min_idx = idxs[dists_masked.index(min(dists_masked))]
                min_val = dists_masked[dists_masked.index(min(dists_masked))]
        else:
            if self.backend=="torch":
                min_val, min_idx = torch.min(dists, dim=0)
                min_idx = min_idx.item()
            elif self.backend=="numpy":
                min_idx = np.argmin(dists)
                min_val = dists[min_idx]
            else:
                min_val = min(dists)
                min_idx = dists.index(min_val)

        return self.to_points()[min_idx], min_idx, float(min_val)

    def boundary_points(self, mat, method: str = "canny", threshold1=100, threshold2=200):
        """
        Given a 2D/3D image array (numpy, torch, or cvMat), extract boundary/edge points.
        Returns a PointCloud
        """
        import cv2
        if isinstance(mat, torch.Tensor):
            mat = mat.cpu().numpy()
        elif 'cv2' in str(type(mat)):
            mat = np.array(mat)

        # For 2D image
        if mat.ndim==2:
            if method=="canny":
                edges = cv2.Canny(mat.astype(np.uint8), threshold1, threshold2)
            else:
                raise GeometryError("Unsupported edge detection method")
            points_idx = np.argwhere(edges>0)
            points = [ [x,y] for y,x in points_idx ]  # flip for x,y
            return PointCloud(points, backend="numpy")
        elif mat.ndim==3:  # 3D array
            # Basic approach: detect non-zero voxels on boundary
            from scipy.ndimage import binary_erosion
            mask = mat>0
            eroded = binary_erosion(mask)
            boundary_mask = mask & (~eroded)
            coords = np.argwhere(boundary_mask)
            return PointCloud(coords, backend="numpy")
        else:
            raise GeometryError("Unsupported input shape")

    def move_to_origin(self):
        """Translate the point cloud so that its centroid is at origin"""
        if self.is_empty():
            return
        center = (self._mins + self._maxs)/2 if self.backend in ["numpy","torch"] else [(self._mins[i]+self._maxs[i])/2 for i in range(self.dim)]
        self.offset(*[-c for c in center])

    def center_to(self, point: 'Point'):
        """Translate the point cloud so that its centroid is at the given point"""
        if self.is_empty():
            return
        target = point.to_numpy() if self.backend=="numpy" else point.to_tensor(device=self.device) if self.backend=="torch" else point.to_list()
        centroid = (self._mins + self._maxs)/2 if self.backend in ["numpy","torch"] else [(self._mins[i]+self._maxs[i])/2 for i in range(self.dim)]
        delta = target - centroid if self.backend in ["numpy","torch"] else [target[i]-centroid[i] for i in range(self.dim)]
        self.offset(*delta)
        
    def offset(self, x=0.0, y=0.0, z=0.0):
        """Translate by given offset"""
        vec = [x,y] if self.dim==2 else [x,y,z]
        if self.is_empty():
            return
        if self.backend=="numpy":
            self.data += np.array(vec)
        elif self.backend=="torch":
            self.data += torch.tensor(vec, dtype=self.data.dtype, device=self.device)
        else:
            self.data = [[p[i]+vec[i] for i in range(self.dim)] for p in self.data]
        self._recalculate_bounds()

    def regularize(self, method="linear", step=None):
        """
        Fill in missing points based on method
        method: "linear", "spline", "cubic"
        step: distance step between consecutive points
        """
        if self.is_empty() or len(self.data)<=1:
            return

        import numpy as np
        from scipy.interpolate import interp1d

        data_arr = self.to_numpy() if self.backend=="numpy" else np.array(self.to_list())
        N = data_arr.shape[0]

        # compute cumulative distance
        diffs = np.diff(data_arr, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumdist = np.concatenate([[0], np.cumsum(dists)])

        # define new positions
        if step is None:
            step = np.min(dists)/2
        new_dist = np.arange(0, cumdist[-1], step)

        if method=="linear":
            f = interp1d(cumdist, data_arr, axis=0)
            new_data = f(new_dist)
        else:
            f = interp1d(cumdist, data_arr, kind=method, axis=0)
            new_data = f(new_dist)

        self.data = new_data if self.backend=="numpy" else torch.tensor(new_data, dtype=self.data.dtype, device=self.device) if self.backend=="torch" else new_data.tolist()
        self._recalculate_bounds()

    def smooth(self, method="linear", window=3):
        """
        Smooth the point cloud using interpolation or moving average
        method: "linear", "spline", "cubic", "moving_average"
        """
        if self.is_empty() or len(self.data)<=1:
            return
        import numpy as np
        data_arr = self.to_numpy() if self.backend=="numpy" else np.array(self.to_list())
        N = data_arr.shape[0]

        if method=="moving_average":
            kernel = np.ones(window)/window
            for dim in range(data_arr.shape[1]):
                data_arr[:,dim] = np.convolve(data_arr[:,dim], kernel, mode='same')
        else:
            # spline or cubic interpolation
            from scipy.interpolate import interp1d
            x = np.arange(N)
            f = interp1d(x, data_arr, kind=method, axis=0)
            data_arr = f(x)

        self.data = data_arr if self.backend=="numpy" else torch.tensor(data_arr, dtype=self.data.dtype, device=self.device) if self.backend=="torch" else data_arr.tolist()
        self._recalculate_bounds()

    def compress(self):
        """Compress the point cloud by converting to float32"""
        if self.is_empty() or self.backend=="plain":
            return
        if self.backend=="numpy":
            self.data = self.data.astype(np.float32)
        elif self.backend=="torch":
            self.data = self.data.to(torch.float32)

    def decompress(self):
        """Decompress point cloud back to default dtype"""
        if self.is_empty() or self.backend=="plain":
            return
        if self.backend=="numpy":
            self.data = self.data.astype(self.dtype)
        elif self.backend=="torch":
            torch_dtype = getattr(torch, self.dtype, torch.float64)
            self.data = self.data.to(torch_dtype)

    def transpose(self):
        """Transpose point cloud array (swap axes)"""
        if self.is_empty() or self.backend=="plain":
            if self.backend=="plain":
                self.data = list(map(list, zip(*self.data)))
            return
        self.data = self.data.T
        self._recalculate_bounds()

    def compress_advanced(self, method="linear", tol=1e-3):
        """
        Compress the point cloud by removing redundant points.
        - method: "linear", "spline", "cubic", "bilinear", etc.
        - tol: tolerance for linear approximation
        Returns metadata required for decompression.
        """
        if self.is_empty() or len(self.data)<=2:
            return None

        # Convert to numpy for simplicity in processing
        data_arr = self.to_numpy() if self.backend=="numpy" else np.array(self.to_list())

        if method=="linear":
            # Douglas-Peucker like simplification for straight lines
            from scipy.spatial import distance

            def dp_recursive(points, tol):
                """Recursive Douglas-Peucker simplification"""
                if len(points)<=2:
                    return [0, len(points)-1]
                start, end = points[0], points[-1]
                line_vec = end - start
                line_len2 = np.sum(line_vec**2)
                if line_len2==0:
                    dists = np.linalg.norm(points - start, axis=1)
                else:
                    t = np.dot(points-start, line_vec) / line_len2
                    t = np.clip(t,0,1)
                    proj = start + t[:,None]*line_vec
                    dists = np.linalg.norm(points - proj, axis=1)
                max_dist_idx = np.argmax(dists)
                if dists[max_dist_idx]<=tol:
                    return [0, len(points)-1]
                else:
                    left = dp_recursive(points[:max_dist_idx+1], tol)
                    right = dp_recursive(points[max_dist_idx:], tol)
                    return left[:-1] + [i+max_dist_idx for i in right]

            key_indices = dp_recursive(data_arr, tol)
            compressed_data = data_arr[key_indices]
            metadata = {"method": method, "indices": key_indices, "original_len": len(data_arr)}
        
        else:
            # For spline, cubic, bilinear
            from scipy.interpolate import interp1d
            N = len(data_arr)
            x = np.arange(N)
            f = interp1d(x, data_arr, kind=method, axis=0)
            step = max(2, N//10)  # keep 10 control points
            key_indices = np.arange(0,N,step)
            compressed_data = data_arr[key_indices]
            metadata = {"method": method, "indices": key_indices, "original_len": N}

        # Update point cloud
        if self.backend=="numpy":
            self.data = compressed_data
        elif self.backend=="torch":
            self.data = torch.tensor(compressed_data, dtype=self.data.dtype, device=self.device)
        else:
            self.data = compressed_data.tolist()

        self._recalculate_bounds()
        return metadata

    def decompress_advanced(self, metadata):
        """
        Decompress a previously compressed point cloud using the metadata.
        """
        if self.is_empty() or metadata is None:
            return

        method = metadata.get("method")
        key_indices = metadata.get("indices")
        original_len = metadata.get("original_len")

        if method=="linear":
            # Linear interpolation between key points
            if self.backend=="numpy":
                x_key = key_indices
                y_key = self.data
                x_full = np.arange(original_len)
                data_full = np.zeros((original_len, self.data.shape[1]))
                for dim in range(self.data.shape[1]):
                    data_full[:,dim] = np.interp(x_full, x_key, y_key[:,dim])
                self.data = data_full
            elif self.backend=="torch":
                x_key = torch.tensor(key_indices, device=self.device, dtype=self.data.dtype)
                y_key = self.data
                x_full = torch.arange(original_len, device=self.device, dtype=self.data.dtype)
                data_full = torch.zeros((original_len, self.data.shape[1]), device=self.device, dtype=self.data.dtype)
                for dim in range(self.data.shape[1]):
                    data_full[:,dim] = torch.interp(x_full, x_key, y_key[:,dim])
                self.data = data_full
            else:
                # plain Python: linear interpolation
                new_data = []
                key_pts = self.data
                for i in range(len(key_pts)-1):
                    start, end = key_pts[i], key_pts[i+1]
                    steps = key_indices[i+1] - key_indices[i]
                    for s in range(steps):
                        t = s/steps
                        new_pt = [start[d]*(1-t)+end[d]*t for d in range(len(start))]
                        new_data.append(new_pt)
                new_data.append(key_pts[-1])
                self.data = new_data
        else:
            # spline/cubic interpolation
            import numpy as np
            from scipy.interpolate import interp1d
            x_key = key_indices
            y_key = self.to_numpy() if self.backend=="numpy" else np.array(self.to_list())
            x_full = np.arange(original_len)
            f = interp1d(x_key, y_key, kind=method, axis=0)
            data_full = f(x_full)
            if self.backend=="numpy":
                self.data = data_full
            elif self.backend=="torch":
                self.data = torch.tensor(data_full, dtype=self.data.dtype, device=self.device)
            else:
                self.data = data_full.tolist()

        self._recalculate_bounds()

