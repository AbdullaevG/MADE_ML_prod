"""Features with outliers and nulls"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class OutliersNulls:
    """ Structure contain parameters for preparing data """
    outliers: str = field(default= 'RestingBP')
    nulls: str = field(default= 'Cholesterol')
    target: str = field(default='HeartDisease')
