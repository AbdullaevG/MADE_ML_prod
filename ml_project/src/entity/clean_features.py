"""Features with outliers and nulls"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class OutliersNulls:
    """ Structure contain parameters for preparing data """
    nulls: str = field(default='TotalCharges')
    outliers: str = field(default= 'TotalCharges')
    target: str = field(default='Churn')
