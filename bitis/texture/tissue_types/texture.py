from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np
import pandas as pd


@dataclass
class Texture:
    # Base array notation:
    # 0 - healthy
    # 1 - fibrosis
    name: str
    tissue: np.ndarray
    boundary: Optional[np.ndarray] = None
    dim: int
    
    properties: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tissue, np.ndarray):
            raise TypeError("tissue should be a numpy ndarray.")
        
    def as_triple_notation(self):
        """Return the texture array with specified notations."""
        triple_tissue = np.zeros(self.tissue.shape)
        triple_tissue = np.where(self.tissue == 1, 2, triple_tissue)
        triple_tissue = np.where(self.tissue == 0, 1, triple_tissue)
        triple_tissue[self.boundary] = 0
        return triple_tissue
    
    def tp(self):
        return self.as_triple_notation()
        
    def as_dataframe(self, properties_only: bool = False):
        """Return a pandas DataFrame representation of the texture array and properties."""        
        # Convert properties to DataFrame format
        if properties_only:
            return pd.DataFrame({
                'Name': [self.name],
                **self.properties  # Include any extra properties
            })
        else:
            return pd.DataFrame({
                'Name': [self.name],
                'Dimension': [self.dim],
                'Tissue': [self.tissue],
                'Boundary': [self.boundary],
                **self.properties  # Include any extra properties
            })

    def as_dict(self):
        """Return a dictionary representation of the texture and its properties."""
        return {
            'name': self.name,
            'dimension': self.dim,
            'tissue': self.tissue.tolist(),  # Converting numpy array to list
            'boundary': self.boundary.tolist() if self.boundary is not None else None,
            **self.properties  # Include any extra properties
        }

