"""The two-head network (shared backbone + classification head + regression head),
with its training procedures and its evaluation metrics, all as methods of one class.

This file only DEFINES things (no code runs when it is imported).
Use it from another file with:   from make_model import TwoHeadNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time 


def make_mlp(in_features, hidden, out_features, dropout=None, final_activation=None):
    """Builds Linear-ReLU(-Dropout) blocks for each hidden size, then a final Linear.

    hidden  : list of hidden layer sizes, e.g. [64, 64]
    dropout : list of dropout probabilities, one per hidden layer (0 = no dropout)

    example: make_mlp(120, [64, 64], 2, [0.25, 0.5], final_activation=nn.Softmax(dim=1))
    creates a network with 120 inputs, two hidden layers of size 64, and 2 outputs,
    with ReLU activations. 
    """
    if dropout is None:
        dropout = [0.0] * len(hidden)
    assert len(dropout) == len(hidden), "dropout must have one entry per hidden layer"

    layers, prev = [], in_features
    for h, p in zip(hidden, dropout):
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if p > 0:
            layers.append(nn.Dropout(p))
        prev = h
    layers.append(nn.Linear(prev, out_features))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class TwoHeadNet(nn.Module): # This class inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
    """Shared backbone + classification head (superconductor or not) + regression head (Tc).
    The default arguments reproduce a simple two-head network with one layer for each part, each layer with 16 neurons.

    Labels y have 3 columns: [is NOT superconductor, is superconductor, Tc].
    """

    # ------------------------------------------------------------------ structure
    def __init__(self,
             in_features=120,
             backbone=(16,), backbone_dropout=(0,),
             cls_hidden=(16,), cls_dropout=(0,),
             reg_hidden=(16,), reg_dropout=(0,)):
        
        super().__init__()
        
        self.config = dict(in_features=in_features,
                       backbone=tuple(backbone), backbone_dropout=tuple(backbone_dropout),
                       cls_hidden=tuple(cls_hidden), cls_dropout=tuple(cls_dropout),
                       reg_hidden=tuple(reg_hidden), reg_dropout=tuple(reg_dropout))

        # shared backbone: Flatten + (Linear, ReLU) for each size in `backbone`
        assert len(backbone_dropout) == len(backbone), "backbone_dropout must have one entry per backbone layer"
        layers, prev = [nn.Flatten()], in_features
        for h, p in zip(backbone, backbone_dropout):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if p > 0:
                layers.append(nn.Dropout(p))
            prev = h

        self.backbone = nn.Sequential(*layers)

        self.cls_head = make_mlp(prev, list(cls_hidden), 2, list(cls_dropout),
                                 final_activation=nn.Softmax(dim=1))
        self.reg_head = make_mlp(prev, list(reg_hidden), 1, list(reg_dropout))

    def forward(self, x):
        z = self.backbone(x)                              # x must be of shape (N, 1, 10, 12) or (N, 120)
        return [self.reg_head(z), self.cls_head(z)]       # [Tc, class probabilities]

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ save / load
    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path)
        model = cls(**d["config"])
        model.load_state_dict(d["state_dict"])
        return model

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def predict(self, ds, batch_size=1024):
        """Returns (Tc predictions, class probabilities, true labels) as CPU double tensors."""
        p0 = next(self.parameters())
        self.eval()                                         # dropout off
        tcs, probs, ys = [], [], []
        for x, y in DataLoader(ds, batch_size=batch_size):
            tc, p = self(x.to(p0.device, p0.dtype))
            tcs.append(tc[:, 0].cpu())
            probs.append(p.cpu())
            ys.append(y)
        return torch.cat(tcs).double(), torch.cat(probs).double(), torch.cat(ys).double()

    def evaluate(self, ds):
        """Quick check: (MAE of Tc in kelvin on the superconductors, classification accuracy)."""
        tc, p, y = self.predict(ds)
        is_sc = y[:, 1] > 0.5
        mae = (tc[is_sc] - y[is_sc, 2]).abs().mean().item()
        acc = (p.argmax(1) == y[:, :2].argmax(1)).double().mean().item()
        return mae, acc

    def scores(self, ds, train_ds):
        """Full metrics on `ds`. `train_ds` gives the trivial baselines (mean Tc, majority class)."""
        tc, p, y = self.predict(ds)
        is_sc = y[:, 1] > 0.5
        err = tc[is_sc] - y[is_sc, 2]

        y_tr = train_ds.tensors[1].double()
        sc_tr = y_tr[:, 1] > 0.5
        tc_mean = y_tr[sc_tr, 2].mean().item()              # baseline of R2: mean Tc of the training superconductors
        p_sc = sc_tr.double().mean().item()
        p_major = max(p_sc, 1 - p_sc)                       # baseline of accuracy: always the majority class

        sse = (err ** 2).sum().item()
        sst = ((y[is_sc, 2] - tc_mean) ** 2).sum().item()
        acc = (p.argmax(1) == y[:, :2].argmax(1)).double().mean().item()
        r2 = 1 - sse / sst
        acc_skill = (acc - p_major) / (1 - p_major)
        return {"MAE_K": err.abs().mean().item(), "RMSE_K": err.pow(2).mean().sqrt().item(),
                "R2": r2, "acc": acc, "acc_skill": acc_skill,
                "overall": 0.5 * (r2 + acc_skill)}

    @staticmethod
    def relative_score(cur, ref):
        """cur, ref: dicts returned by scores(); ref = reference network on the SAME dataset."""
        reg = 1 - (cur["RMSE_K"] / ref["RMSE_K"]) ** 2      # 1 - MSE/MSE_ref
        cls = 1 - (1 - cur["acc"]) / (1 - ref["acc"])        # relative error reduction
        return {"reg": reg, "cls": cls, "overall": 0.5 * (reg + cls)}

    # ------------------------------------------------------------------ training
    @staticmethod
    def _reg_loss(pred_tc, y, reg_on, reduction, tc_scale=1.0):
        """MSE on Tc. reg_on: "all" -> all samples; "sc" -> only the superconductors."""
        if reg_on == "sc":
            m = y[:, 1] > 0.5
            if not m.any():
                return pred_tc.sum() * 0.0                  # batch without superconductors
            pred_tc, tc = pred_tc[m], y[m, 2]
        else:
            tc = y[:, 2]
        return F.mse_loss(pred_tc / tc_scale, tc / tc_scale, reduction=reduction)

    def _run(self, stage, params, epochs, step_fn, train_dl, val_ds, lr, lr_drop_epoch,
             lr_after_drop, device, dtype, eval_every, hist, reduction):
        """Generic training loop: Adam, learning rate drop, logging. `step_fn(x, y)` returns the loss."""
        opt = torch.optim.Adam(params, lr=lr)
        for epoch in range(1, epochs + 1):
            if epoch == lr_drop_epoch + 1:                  # learning rate drops after `lr_drop_epoch` epochs
                for g in opt.param_groups:
                    g["lr"] = lr_after_drop
            self.train()
            total, count = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device, dtype), y.to(device, dtype)
                loss = step_fn(x, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * (len(x) if reduction == "mean" else 1)   # -> sum over samples
                count += len(x)
                
            if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
                s = self.scores(val_ds, train_dl.dataset)       # train_dl.dataset = the training set
                hist["stage"].append(stage)
                hist["epoch"].append(epoch)
                hist["train_loss"].append(total / count)
                hist["val_mae_K"].append(s["MAE_K"])
                hist["val_acc"].append(s["acc"])
                hist["val_R2"].append(s["R2"])
                hist["val_acc_skill"].append(s["acc_skill"])
                hist["val_overall"].append(s["overall"])
                hist["time_s"].append(time.perf_counter() - self._t0)
                print(f"[{stage}] epoch {epoch}/{epochs}  loss {total/count:.4f}  "
                      f"MAE {s['MAE_K']:.2f} K  R2 {s['R2']:.3f}  acc {s['acc']:.3f}", flush=True)



    @staticmethod
    def _new_history():
        return {k: [] for k in ("stage", "epoch", "train_loss", "val_mae_K", "val_acc",
                            "val_R2", "val_acc_skill", "val_overall", "time_s")}

    def train_two_stage(self, train_ds, val_ds, epochs_reg=5000, epochs_cls=5000, batch_size=64,
                      lr=1e-4, lr_drop_epoch=3000, lr_after_drop=1e-5, reduction="sum",
                      reg_on="all", dtype=torch.float64, device=None, eval_every=10):
        """Training of Kaplan et al. (defaults = values used in the paper's code).

        Stage 1: backbone + regression head, MSE on Tc.
        Stage 2: backbone and regression head frozen, classification head trained with the MSE
                 between the softmax output and the one-hot labels.
        In each stage the learning rate drops from `lr` to `lr_after_drop` at epoch `lr_drop_epoch`.
        reg_on: "all" -> stage 1 uses all the samples (non-superconductors have Tc = 0, as in the paper);
                "sc"  -> stage 1 uses only the superconductors.
        Returns a dict with the history.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device, dtype=dtype)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        hist = self._new_history()
        self._t0 = time.perf_counter()
        common = dict(train_dl=train_dl, val_ds=val_ds, lr=lr, lr_drop_epoch=lr_drop_epoch,
                      lr_after_drop=lr_after_drop, device=device, dtype=dtype,
                      eval_every=eval_every, hist=hist, reduction=reduction)

        # ---------- Stage 1: Tc regression ----------
        def reg_step(x, y):
            pred = self.reg_head(self.backbone(x))[:, 0]
            return self._reg_loss(pred, y, reg_on, reduction)

        self._run("stage 1", list(self.backbone.parameters()) + list(self.reg_head.parameters()),
                  epochs_reg, reg_step, **common)

        # ---------- Stage 2: classification ----------
        frozen = list(self.backbone.parameters()) + list(self.reg_head.parameters())
        for p in frozen:
            p.requires_grad = False

        def cls_step(x, y):
            with torch.no_grad():
                z = self.backbone(x)
            return F.mse_loss(self.cls_head(z), y[:, :2], reduction=reduction)

        self._run("stage 2", list(self.cls_head.parameters()), epochs_cls, cls_step, **common)

        for p in frozen:
            p.requires_grad = True
        return hist

    def train_joint(self, train_ds, val_ds, epochs=5000, batch_size=64,
                  lr=1e-4, lr_drop_epoch=3000, lr_after_drop=1e-5,
                  cls_weight=1.0, tc_scale=1.0, reduction="mean", reg_on="sc",
                  dtype=torch.float64, device=None, eval_every=10):
        """Trains the whole network at once, with  loss = MSE(Tc) + cls_weight * cross-entropy.

        tc_scale: Tc is divided by this number inside the MSE, so that the two terms have similar size
                  (e.g. 100 for Tc in kelvin). cls_weight: relative weight of the classification term.
        reg_on:   "sc" -> MSE only on the superconductors (default); "all" -> on all the samples.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device, dtype=dtype)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        hist = self._new_history()
        self._t0 = time.perf_counter()

        def joint_step(x, y):
            z = self.backbone(x)
            reg = self._reg_loss(self.reg_head(z)[:, 0], y, reg_on, reduction, tc_scale)
            probs = self.cls_head(z)                                   # softmax output
            target = y[:, :2].argmax(1)
            ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), target, reduction=reduction)
            return reg + cls_weight * ce

        self._run("joint", list(self.parameters()), epochs, joint_step, train_dl, val_ds, lr,
                  lr_drop_epoch, lr_after_drop, device, dtype, eval_every, hist, reduction)
        return hist