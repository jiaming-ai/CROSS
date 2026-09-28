import torch
from collections import defaultdict
from typing import Dict, List, Tuple


class SpatialHashGrid3D:
    """
    Simple 3D spatial hash grid for proximity queries.

    Stores points in cubic cells of size `cell_size` and allows querying points
    in the 3x3x3 neighborhood around a target position. Intended for small
    point clouds where full KD-trees are unnecessary.
    """

    def __init__(self, cell_size: float):
        self.cell_size = float(cell_size)
        self.grid: Dict[Tuple[int, int, int], List[Tuple[int, torch.Tensor]]] = defaultdict(list)

    def _cell_from_pos(self, pos: torch.Tensor) -> Tuple[int, int, int]:
        """Compute integer cell coordinates for a 3D position tensor."""
        pos_cpu = pos.detach().cpu().view(-1)[:3]
        cell = torch.floor(pos_cpu / self.cell_size).to(torch.int64)
        return tuple(cell.tolist())

    def insert(self, key: int, pos: torch.Tensor) -> None:
        """Insert a point into the grid."""
        cell = self._cell_from_pos(pos)
        self.grid[cell].append((key, pos))

    @classmethod
    def from_positions(cls, positions: Dict[int, torch.Tensor], cell_size: float) -> "SpatialHashGrid3D":
        """Build a grid from a dict of {id: position}."""
        grid = cls(cell_size)
        for k, p in positions.items():
            grid.insert(k, p)
        return grid

    def query(self, pos: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
        """Return all points in the 3x3x3 neighborhood of the cell containing `pos`."""
        cx, cy, cz = self._cell_from_pos(pos)
        neighbors: List[Tuple[int, torch.Tensor]] = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbors.extend(self.grid.get((cx + dx, cy + dy, cz + dz), []))

        return neighbors
