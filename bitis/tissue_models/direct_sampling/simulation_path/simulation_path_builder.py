from abc import ABC, abstractmethod


class SimulationPathBuilder(ABC):
    """
    A class that builds simulation paths.
    """

    def __init__(self):
        pass

    @abstractmethod
    def build(self):
        pass
