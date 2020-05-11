try:
    from sklearn.cross_validation import KFold
except ImportError:
    import numpy as np
    from sklearn.model_selection import KFold as _KFold

    class KFold(object):
        def __init__(self, n, n_folds, **kwargs):
            self.kf = _KFold(n_splits=n_folds, **kwargs)
            self.n = n

        def __iter__(self):
            yield from self.kf.split(np.arange(self.n))
