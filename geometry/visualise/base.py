from abc import ABC, abstractmethod

class Renderer(ABC):
    """
    Base renderer contract.
    """

    name: str

    @abstractmethod
    def draw_point(self, point, **style):
        raise NotImplementedError

    @abstractmethod
    def draw_line(self, line, **style):
        raise NotImplementedError

    @abstractmethod
    def draw_polygon(self, polygon, **style):
        raise NotImplementedError
