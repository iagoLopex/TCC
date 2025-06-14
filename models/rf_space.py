# models/rf_space.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import config

class RandomForestSpace:
    """Define a arquitetura, espaço de busca e avaliação para o modelo Random Forest."""
    n_estimators_opts = [50, 100, 200, 300]
    max_depth_opts = [5, 10, 20, None]
    min_samples_split_opts = [2, 5, 10]
    min_samples_leaf_opts = [1, 2, 4]
    max_features_opts = ['sqrt', 'log2', None]
    
    bounds = np.array([[0, len(n_estimators_opts)-1], [0, len(max_depth_opts)-1],
                       [0, len(min_samples_split_opts)-1], [0, len(min_samples_leaf_opts)-1],
                       [0, len(max_features_opts)-1]])
    types = np.array(['int', 'int', 'int', 'int', 'int'])

    def __init__(self, Xtr, ytr, Xv, yv, use_weights=False, **kwargs):
        self.Xtr, self.ytr = Xtr, ytr.ravel()
        self.Xv, self.yv = Xv, yv.ravel()
        self.seed0 = config.SEED
        if use_weights:
            self.w_tr = np.where(self.ytr > kwargs['thr'], kwargs['w_minor'], kwargs['w_major']).astype(np.float32)
        else:
            self.w_tr = np.ones_like(self.ytr)

    @staticmethod
    def decode(g):
        return dict(n_estimators=RandomForestSpace.n_estimators_opts[int(g[0])],
                    max_depth=RandomForestSpace.max_depth_opts[int(g[1])],
                    min_samples_split=RandomForestSpace.min_samples_split_opts[int(g[2])],
                    min_samples_leaf=RandomForestSpace.min_samples_leaf_opts[int(g[3])],
                    max_features=RandomForestSpace.max_features_opts[int(g[4])])

    def _train_model(self, cfg, rep):
        model = RandomForestRegressor(**cfg, random_state=self.seed0 + rep, n_jobs=-1)
        model.fit(self.Xtr, self.ytr, sample_weight=self.w_tr)
        return model, None, None  # Retorna None para os históricos de loss

    def evaluate(self, cfg, rep):
        model, _, _ = self._train_model(cfg, rep)
        r2 = r2_score(self.yv, model.predict(self.Xv))
        return -r2