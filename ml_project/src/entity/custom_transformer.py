""" Custom transformer """

from dataclasses import dataclass


@dataclass
class CustomTransformer:
    """ Structure contain switch for custom transformer """
    use_custom_transformer: bool