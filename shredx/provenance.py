import numpy as np


def get_meta(self):
    import ngmix
    import shredder
    import fofx
    from .version import __version__

    dt = [
        ('ngmix_vers', 'S10'),
        ('shredder_vers', 'S10'),
        ('fofx_vers', 'S10'),
        ('shredx_vers', 'S10'),
    ]
    meta = np.zeros(1, dtype=dt)
    meta['ngmix_vers'] = ngmix.__version__
    meta['shredder_vers'] = shredder.__version__
    meta['fofx_vers'] = fofx.__version__
    meta['shredx_vers'] = __version__

    return meta
