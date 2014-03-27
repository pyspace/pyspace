""" Type conversion helper functions """

import warnings
import yaml

def python2yaml(value):
    """ Conversion function to handle yaml specialties"""
    # due to ugly incompatbilities in the conversion of
    # float (yaml) -> float (python) -> float(yaml) and
    # iterable(python) -> iterable(yaml),
    # we have to introduce this special treatment
    str_representation = yaml.dump([value], default_flow_style=True).strip()
    if str_representation.startswith('-'):
        return str_representation[2:]
    elif str_representation.startswith('['):
        return str_representation[1:-1]
    else:
        warnings.warn("Wrong format for yaml conversion of template (%s)."
                        % str_representation)
        return str(value)
