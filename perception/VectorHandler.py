from dataclasses import dataclass

@dataclass
class WorldPoint:
    x_lateral: float
    z_forward: float


    def __sub__(self, other):
        if not isinstance(other, WorldPoint):
            raise ValueError("Cannot subtract different class types")
        return WorldPoint(self.x_lateral - other.x_lateral, self.z_forward - other.z_forward)

    def __add__(self, other):
        if not isinstance(other, WorldPoint):
            raise ValueError("Cannot add different class types")
        return WorldPoint(self.x_lateral + other.x_lateral, self.z_forward + other.z_forward)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError("Allowed multiplication only by a scalar")
        return WorldPoint(self.x_lateral * scalar, self.z_forward * scalar)

    __rmul__ = __mul__

    def distance(self):
        return (self.x_lateral ** 2 + self.z_forward ** 2) ** 0.5


    