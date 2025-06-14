# models/mlp_space.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
import config

class MLPBlockSpace:
    """Define a arquitetura, espaço de busca e avaliação para o modelo MLP."""
    n_layers = list(range(1, 10, 1))
    units = list(range(1, 100, 1))
    dropout = [0.0, 0.05, 0.1, 0.15]
    acts = [nn.Tanh, nn.Sigmoid, nn.LogSigmoid]
    bss = list(range(6, 50, 1))
    lr_log10 = [0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002, 0.0002, 0.3, 0.03, 0.003, 0.0003, 0.4, 0.04, 0.004, 0.0004]

    bounds = np.array([[0, len(n_layers)-1], [0, len(units)-1], [0, len(dropout)-1], 
                       [0, len(acts)-1], [0, len(bss)-1], [0, len(lr_log10)-1]])
    types = np.array(['int', 'int', 'int', 'int', 'int', 'int'])

    def __init__(self, Xtr, ytr, Xv, yv, use_weights=False, **kwargs):
        self.Xt = torch.from_numpy(Xtr).to(config.DEVICE)
        self.yt = torch.from_numpy(ytr).to(config.DEVICE)
        self.Xv = torch.from_numpy(Xv).to(config.DEVICE)
        self.yv = torch.from_numpy(yv).to(config.DEVICE)
        self.in_dim = Xtr.shape[1]
        self.dev = config.DEVICE
        self.seed0 = config.SEED

        if use_weights:
            self.wt = torch.from_numpy(np.where(ytr.flatten() > kwargs['thr'], kwargs['w_minor'], kwargs['w_major']).astype(np.float32)).to(config.DEVICE)
            self.wv = torch.from_numpy(np.where(yv.flatten() > kwargs['thr'], kwargs['w_minor'], kwargs['w_major']).astype(np.float32)).to(config.DEVICE)
        else:
            self.wt = torch.ones_like(self.yt).to(config.DEVICE)
            self.wv = torch.ones_like(self.yv).to(config.DEVICE)

    @staticmethod
    def decode(g):
        return dict(n_layers=MLPBlockSpace.n_layers[int(g[0])], units=MLPBlockSpace.units[int(g[1])],
                    drop=MLPBlockSpace.dropout[int(g[2])], act=MLPBlockSpace.acts[int(g[3])],
                    batch=MLPBlockSpace.bss[int(g[4])], lr=MLPBlockSpace.lr_log10[int(g[5])])

    def _build(self, cfg):
        layers = []
        in_f = self.in_dim
        drop_block = np.random.randint(0, cfg['n_layers']) if cfg['n_layers'] > 0 else -1
        for idx in range(cfg['n_layers']):
            layers.extend([nn.Linear(in_f, cfg['units']), nn.BatchNorm1d(cfg['units']), cfg['act']()])
            if idx == drop_block and cfg['drop'] > 0: layers.append(nn.Dropout(cfg['drop']))
            in_f = cfg['units']
        layers.append(nn.Linear(in_f, 1))
        return nn.Sequential(*layers).to(self.dev)

    def _train_model(self, cfg, rep):
        torch.manual_seed(self.seed0 + rep)
        model = self._build(cfg)
        opt = optim.Adam(model.parameters(), lr=cfg['lr'])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.Xt, self.yt, self.wt.unsqueeze(1)),
            batch_size=cfg['batch'], shuffle=True, drop_last=True
        )
        train_hist, val_hist = [], []
        wait = 0
        for epoch in range(100):
            model.train()
            epoch_train_loss = []
            for xb, yb, wb in loader:
                opt.zero_grad()
                pred = model(xb)
                loss = (torch.abs(pred - yb) * wb).mean()
                loss.backward()
                opt.step()
                epoch_train_loss.append(loss.item())
            train_hist.append(np.mean(epoch_train_loss))

            model.eval()
            with torch.no_grad():
                v_loss = (torch.abs(model(self.Xv) - self.yv) * self.wv.unsqueeze(1)).mean().item()
                val_hist.append(v_loss)
            
            if epoch > 10 and (val_hist[-1] > val_hist[-2]):
                wait += 1
            else:
                wait = 0
            if wait >= 5: break
        return model, train_hist, val_hist

    def evaluate(self, cfg, rep):
        model, _, _ = self._train_model(cfg, rep)
        with torch.no_grad():
            y_pred_val = model(self.Xv)
            r2 = r2_score(self.yv.cpu().numpy(), y_pred_val.cpu().numpy())
        return -r2