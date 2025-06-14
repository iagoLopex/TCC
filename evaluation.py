# evaluation.py
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import config

def evaluate_gene_configuration(cfg, rep, X_dev, y_dev, X_train_s, y_train_s, X_valid_s, y_valid_s):
    """Função de avaliação que lida com Holdout ou K-Fold."""
    space_class = config.MODEL_CLASSES[config.MODEL_TO_OPTIMIZE]
    params = {'use_weights': config.USE_WEIGHTS, **config.WEIGHT_PARAMS}

    if config.VALIDATION_METHOD == 'Holdout':
        # Para Holdout, os dados já vêm divididos e escalonados
        space = space_class(X_train_s, y_train_s, X_valid_s, y_valid_s, **params)
        return space.evaluate(cfg, rep)

    elif config.VALIDATION_METHOD == 'K-fold':
        kf = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
        fold_scores = []
        for train_idx, val_idx in kf.split(X_dev, y_dev):
            X_train_f, X_val_f = X_dev[train_idx], X_dev[val_idx]
            y_train_f, y_val_f = y_dev[train_idx], y_dev[val_idx]

            scX = StandardScaler().fit(X_train_f)
            X_train_sf, X_val_sf = scX.transform(X_train_f), scX.transform(X_val_f)
            scY = StandardScaler().fit(y_train_f)
            y_train_sf, y_val_sf = scY.transform(y_train_f), scY.transform(y_val_f)

            space_fold = space_class(X_train_sf, y_train_sf, X_val_sf, y_val_sf, **params)
            fold_scores.append(space_fold.evaluate(cfg, rep))
        return np.mean(fold_scores)

class GenericSpace:
    """Classe que atua como uma interface entre o GA e os modelos."""
    def __init__(self, model_type, **kwargs):
        self.base_class = config.MODEL_CLASSES[model_type]
        self.bounds, self.types = self.base_class.bounds, self.base_class.types
        self.eval_kwargs = kwargs # Armazena os dados para passar para a avaliação

    def decode(self, gene):
        return self.base_class.decode(gene)

    def evaluate(self, cfg, rep):
        # Passa os dados armazenados para a função de avaliação
        return evaluate_gene_configuration(cfg, rep, **self.eval_kwargs)