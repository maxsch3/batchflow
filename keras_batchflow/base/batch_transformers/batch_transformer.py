from abc import ABC, abstractmethod


class BatchTransformer(ABC):
    """
    This is an abstract class that defines basic functionality and interfaces of all BatchTransformers
    """
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, batch):
        pass

    @abstractmethod
    def inverse_transform(self, batch):
        pass
