__all__ = ('CONSTANTS', 'set_constant', 'check_constants')

CONSTANTS = {
    'PREFIX': None,
    'DATA_LOAD_FROM': None,
    'RANDOM_STATE': None,
}


def set_constant(constants):
    CONSTANTS.update(constants)

def check_constants():
    for k, v in CONSTANTS.items():
        if v is None:
            raise ValueError(f'{k} is None')
