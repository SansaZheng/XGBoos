"""Knob-XGBoost: the knowledge-boundary / "knob" method of Huang, Mao, Jiang & Cambria
("Uncovering the Knowledge Boundaries and Mechanistic 'Knob' of LLMs") carried over to XGBoost.

Usage: python -m baselines.knob_xgb --dataset sapienza|sd-stock|sd-crypto [--variant raw|causal]

Mapping (paper -> here)
  hidden states H(x) in R^{L x d}  -> staged contribution map: H[l, j] = TreeSHAP contribution of
                                      feature j from the trees of boosting block l (L blocks).
                                      margin(x) = bias + sum_{l,j} H[l, j]  (checked numerically)
  label "answer correct"           -> the alert decision of XGBoost is correct (TP/TN) or not (FN/FP),
                                      measured out-of-fold on the training sequences
  boundary g(H)=0                  -> multi-view CNN probe (vertical / horizontal / local) + LR, MLP
  knobs J_k                        -> top-k features by s_j = sum_l |dA_lj| of the vertical CNN map;
                                      k_knob = first k with Acc(J_k) >= Acc_full - 1 %
  KAN in knob space                -> layer attention + small RBF-KAN Phi(z_bar)
  intervention                     -> shift z_bar by tau / (|grad Phi| + eps) * direction, min-norm
                                      layer patch; because the XGBoost read-out is additive the margin
                                      moves by sum(patch).  Directions: normal, tangent, random.
Everything that is fitted (probe, knobs, KAN) uses training sequences only (out-of-fold);
tau / lambda and the variant are chosen on validation; test labels are used once, for the final metrics.
"""
import argparse
import itertools
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .common import RESULTS, SEEDS, metrics, pick_threshold
from .market import SAPIENZA_FEATURES, causal, load

warnings.filterwarnings('ignore')
torch.set_num_threads(1)
OUT = RESULTS / 'knob'
TAUS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
LAYERS = 8
DELTA = 0.01
CAUSAL_PARTS = ['', '-lag1', '-lag4', '-mean4', '-mean16', 'std16', '-mean64']


# ----------------------------------------------------------------------------- data / model
def design(sequences, variant):
    X = np.vstack([s[0] if variant == 'raw' else causal(s[0]) for s in sequences]).astype(np.float32)
    y = np.concatenate([s[1] for s in sequences])
    group = np.concatenate([np.full(len(s[1]), i) for i, s in enumerate(sequences)])
    return X, y, group


def feature_names(dataset, variant, width):
    base = SAPIENZA_FEATURES + ['log_gap'] if dataset == 'sapienza' else [f'x{i}' for i in range(width)]
    if variant == 'raw':
        return base[:width]
    f = width // 7
    base = base[:f]
    return [(f'{b}{p}' if p != 'std16' else f'std16({b})') for p in CAUSAL_PARTS for b in base]


def xgb_model(seed, pos_weight, rounds=600, early=True):
    return xgb.XGBClassifier(n_estimators=rounds, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=1, scale_pos_weight=pos_weight,
                             tree_method='hist', eval_metric='aucpr',
                             early_stopping_rounds=50 if early else None, random_state=seed, n_jobs=2)


def contribution_map(model, X, trees, layers):
    """H[n, L, d] and margin[n]; block l holds the contributions of trees edges[l]..edges[l+1]."""
    booster = model.get_booster()
    edges = np.unique(np.linspace(0, trees, layers + 1).round().astype(int))
    data = xgb.DMatrix(X)
    # TreeSHAP is additive over trees, so a block's contribution is a difference of cumulative maps
    cum = [np.zeros((X.shape[0], X.shape[1]), dtype=np.float64)] + [
        booster.predict(data, pred_contribs=True, iteration_range=(0, int(b)))[:, :-1].astype(np.float64)
        for b in edges[1:]]
    H = np.stack([b - a for a, b in zip(cum[:-1], cum[1:])], axis=1).astype(np.float32)
    margin = booster.predict(data, output_margin=True, iteration_range=(0, trees)).astype(np.float64)
    bias = margin - H.sum((1, 2))
    assert np.ptp(bias) < 1e-3, f'contribution map is not additive: {np.ptp(bias)}'
    return H, margin


# ----------------------------------------------------------------------------- probes
def cell_weights(y, alert):
    """Balanced weights: incorrect (FN+FP) = 1/2; correct = 1/2 split evenly between TP and TN."""
    correct = (alert == (y == 1)).astype(int)
    w = np.zeros(len(y))
    wrong = correct == 0
    w[wrong] = 0.5 / max(wrong.sum(), 1)
    tp = (correct == 1) & (y == 1); tn = (correct == 1) & (y == 0)
    share = 0.25 if tp.sum() and tn.sum() else 0.5
    if tp.sum(): w[tp] = share / tp.sum()
    if tn.sum(): w[tn] = share / tn.sum()
    return correct, w * len(y)


def balanced_accuracy(correct, score, w):
    pred = score > 0
    return float(sum(w[(correct == c) & (pred == bool(c))].sum() / max(w[correct == c].sum(), 1e-12)
                     for c in (0, 1)) / 2)


class MultiViewCNN(nn.Module):
    def __init__(self, layers, dims, channels=8):
        super().__init__()
        odd = lambda v: max(1, v - (1 - v % 2))
        kv, kh, kl = odd(min(9, layers)), odd(min(33, dims)), odd(min(5, layers, dims))
        self.vertical = nn.Conv2d(1, channels, (kv, 1), padding=(kv // 2, 0))
        self.horizontal = nn.Conv2d(1, channels, (1, kh), padding=(0, kh // 2))
        self.local = nn.Conv2d(1, channels, (kl, kl), padding=kl // 2)
        self.head = nn.Linear(3 * channels, 2)

    def maps(self, x):
        x = x[:, None]
        return [torch.relu(b(x)) for b in (self.vertical, self.horizontal, self.local)]

    def forward(self, x):
        logits = self.head(torch.cat([m.amax((2, 3)) for m in self.maps(x)], 1))
        return logits[:, 1] - logits[:, 0]


class MLP(nn.Module):
    def __init__(self, width, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def fit_torch(model, x, target, w, steps=300, lr=1e-2, decay=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    x = torch.as_tensor(x, dtype=torch.float32); t = torch.as_tensor(target, dtype=torch.float32)
    w = torch.as_tensor(w / w.sum(), dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(model(x), t, reduction='none') * w).sum()
        loss.backward(); opt.step()
    return model.eval()


def scale(train, *others):
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-6
    return [(a - mu) / sd for a in (train,) + others]


def probe(kind, seed, xtr, ctr, wtr, xte):
    """Fit one probe on (xtr, correct, weights); return the boundary score g on xte."""
    xtr, xte = scale(xtr, xte)
    torch.manual_seed(seed)
    if kind == 'lr':
        m = LogisticRegression(C=1.0, max_iter=500).fit(xtr.reshape(len(xtr), -1), ctr, sample_weight=wtr)
        return m.decision_function(xte.reshape(len(xte), -1)), None
    if kind == 'mlp':
        m = fit_torch(MLP(int(np.prod(xtr.shape[1:]))), xtr, ctr, wtr)
    else:
        m = fit_torch(MultiViewCNN(xtr.shape[1], xtr.shape[2]), xtr, ctr, wtr, steps=200)
    with torch.no_grad():
        g = m(torch.as_tensor(xte, dtype=torch.float32)).numpy()
        ranking = None
        if kind == 'cnn':     # coarse localisation: dA = mean map(correct) - mean map(incorrect)
            a = m.maps(torch.as_tensor(xtr, dtype=torch.float32))[0].mean(1).numpy()     # vertical view
            wn = wtr[:, None, None]
            delta = ((a * wn)[ctr == 1].sum(0) / wtr[ctr == 1].sum()
                     - (a * wn)[ctr == 0].sum(0) / wtr[ctr == 0].sum())
            ranking = delta
    return g, ranking


def cv_accuracy(kinds, seed, x, c, w, groups, folds, want_rank=False):
    """Pooled grouped-CV balanced accuracy of the best probe kind; optional per-fold dA maps."""
    scores = {k: np.zeros(len(c)) for k in kinds}
    deltas = []
    for tr, te in folds:
        if len(np.unique(c[tr])) < 2:
            continue
        for k in kinds:
            g, d = probe(k, seed, x[tr], c[tr], w[tr], x[te])
            scores[k][te] = g
            if d is not None: deltas.append(d)
    acc = {k: balanced_accuracy(c, s, w) for k, s in scores.items()}
    return (acc, deltas) if want_rank else acc


# ----------------------------------------------------------------------------- KAN in knob space
class KANLayer(nn.Module):
    """Kolmogorov-Arnold layer: every edge is a learnable 1-D function (RBF expansion + linear term)."""
    def __init__(self, n_in, n_out, grid=8, span=2.5):
        super().__init__()
        self.register_buffer('centres', torch.linspace(-span, span, grid))
        self.width = 2 * span / (grid - 1)
        self.coef = nn.Parameter(torch.zeros(n_out, n_in, grid))
        self.linear = nn.Parameter(torch.randn(n_out, n_in) * 0.3)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        basis = torch.exp(-((x[..., None] - self.centres) / self.width) ** 2)
        return torch.einsum('nig,oig->no', basis, self.coef) + x @ self.linear.T + self.bias


class KnobKAN(nn.Module):
    """g(H) = Phi(sum_l alpha_l H[l, J]) with alpha = softmax(w) and Phi a k-2-1 KAN."""
    def __init__(self, layers, k, mu, sd):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(layers))
        self.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))
        self.l1, self.l2 = KANLayer(k, 2), KANLayer(2, 1)

    def alpha(self):
        return torch.softmax(self.w, 0)

    def pool(self, z):
        return (z * self.alpha()[None, :, None]).sum(1)

    def phi(self, zbar):
        return self.l2(torch.tanh(self.l1((zbar - self.mu) / self.sd)) * 2.0)[:, 0]

    def forward(self, z):
        return self.phi(self.pool(z))


def fit_kan(seed, z, c, w):
    torch.manual_seed(seed)
    pooled = z.mean(1)
    model = KnobKAN(z.shape[1], z.shape[2], pooled.mean(0), pooled.std(0) + 1e-6)
    return fit_torch(model, z, c, w, steps=500, lr=2e-2, decay=1e-4)


def kan_state(model, z, chunk=200_000):
    """Phi, grad Phi (w.r.t. pooled knob coordinates) for every row."""
    phi, grad = [], []
    for b in range(0, len(z), chunk):
        zbar = model.pool(torch.as_tensor(z[b:b + chunk], dtype=torch.float32)).detach().requires_grad_(True)
        out = model.phi(zbar)
        grad.append(torch.autograd.grad(out.sum(), zbar)[0].numpy()); phi.append(out.detach().numpy())
    return np.concatenate(phi).astype(np.float64), np.concatenate(grad).astype(np.float64)


def directions(grad, seed):
    """Unit normal, a unit tangent (orthogonal to the normal) and a random unit direction."""
    rng = np.random.default_rng(seed)
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    normal = grad / (norm + 1e-12)
    out = {'normal': normal}
    rand = rng.standard_normal(grad.shape)
    out['random'] = rand / np.linalg.norm(rand, axis=1, keepdims=True)
    if grad.shape[1] > 1:
        t = rng.standard_normal(grad.shape)
        t -= (t * normal).sum(1, keepdims=True) * normal
        out['tangent'] = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return out, norm[:, 0]


def margin_shift(tau, direction, grad_norm, alpha, eps, clip=20.0):
    """Paper patch dH[l, J] = alpha_l / sum(alpha^2) * dz, dz = tau / (|grad Phi| + eps) * d.
    XGBoost reads every block additively, so the margin moves by sum_l sum_j dH[l, j]."""
    dz = tau / (grad_norm[:, None] + eps) * direction
    return np.clip(alpha.sum() / (alpha ** 2).sum() * dz.sum(1), -clip, clip)


# ----------------------------------------------------------------------------- evaluation helpers
def log_likelihood(y, margin):
    return -np.logaddexp(0, -np.where(y == 1, margin, -margin))


def evaluate(yva, sva, yte, ste):
    thr = pick_threshold(yva, sva)
    return dict(validation=metrics(yva, sva, thr), test=metrics(yte, ste, thr))


def budget_recall(y, score, alerts):
    """Recall when exactly `alerts` rows may be flagged (top scores)."""
    top = np.argsort(-score, kind='stable')[:alerts]
    return float(y[top].sum() / max(y.sum(), 1))


def brief(e):
    t = e['test']
    return (f"val AP={e['validation']['AP']:.4f} | test AP={t['AP']:.4f} F1={t['F1']:.4f} "
            f"R={t['recall']:.3f} TP/FP/FN={t['TP']}/{t['FP']}/{t['FN']}")


# ----------------------------------------------------------------------------- one run
def run(dataset, variant, seed, splits, layers=LAYERS, save=True):
    (Xtr, ytr, gtr), (Xva, yva, _), (Xte, yte, _) = (design(splits[s], variant) for s in
                                                    ('train', 'validation', 'test'))
    names = feature_names(dataset, variant, Xtr.shape[1])
    pos_weight = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))
    rng = np.random.default_rng(seed)

    # 1. the baseline exactly as baselines.market fits it
    model = xgb_model(seed, pos_weight).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    trees = model.best_iteration + 1
    Hva, mva = contribution_map(model, Xva, trees, layers)
    Hte, mte = contribution_map(model, Xte, trees, layers)
    L, d = Hva.shape[1:]
    base = evaluate(yva, mva, yte, mte)
    thr = base['test']['threshold']
    print(f'[{dataset} {variant} seed={seed}] trees={trees} L={L} d={d}  baseline: {brief(base)}', flush=True)

    # 2. out-of-fold hidden states and correctness labels on the training sequences
    Htr = np.zeros((len(ytr), L, d), dtype=np.float32); mtr = np.zeros(len(ytr))
    n_folds = min(5, len(np.unique(gtr)))
    for a, b in GroupKFold(n_folds).split(Xtr, ytr, gtr):
        fold = xgb_model(seed, pos_weight, rounds=trees, early=False).fit(Xtr[a], ytr[a], verbose=False)
        Htr[b], mtr[b] = contribution_map(fold, Xtr[b], trees, layers)
    thr_oof = pick_threshold(ytr, mtr)
    alert_tr = mtr >= thr_oof
    correct_all, _ = cell_weights(ytr, alert_tr)
    keep = (ytr == 1) | (correct_all == 0)
    spare = np.flatnonzero(~keep)
    keep[rng.choice(spare, min(len(spare), 2000), replace=False)] = True     # sampled true negatives
    idx = np.flatnonzero(keep)
    Hp, yp, gp, mp = Htr[idx], ytr[idx], gtr[idx], mtr[idx]
    cp, wp = cell_weights(yp, alert_tr[idx])
    cells = dict(TP=int(((cp == 1) & (yp == 1)).sum()), FN=int(((cp == 0) & (yp == 1)).sum()),
                 FP=int(((cp == 0) & (yp == 0)).sum()), TN_sampled=int(((cp == 1) & (yp == 0)).sum()))
    folds = list(GroupKFold(min(5, len(np.unique(gp)))).split(Hp, cp, gp))

    # 3. RQ1: is the boundary learnable?  full map vs the margin alone (uncertainty control)
    acc_full, deltas = cv_accuracy(('cnn', 'mlp', 'lr'), seed, Hp, cp, wp, gp, folds, want_rank=True)
    unc = np.column_stack([mp, np.abs(mp - thr_oof)])[:, None, :]
    acc_margin = cv_accuracy(('lr', 'mlp'), seed, unc, cp, wp, gp, folds)
    best_full = max(acc_full.values())

    # 4. RQ2/3: coarse-to-fine knobs.  dA from the CNN fitted on the whole probe set gives the ranking;
    #    the per-fold maps give its stability.
    _, delta = probe('cnn', seed, Hp, cp, wp, Hp)
    s = np.abs(delta).sum(0)
    order = np.argsort(-s)
    late_ratio = float(np.abs(delta)[L // 2:].sum() / max(np.abs(delta).sum(), 1e-12))
    fold_top = [list(np.argsort(-np.abs(dl).sum(0))[:2]) for dl in deltas]
    acc_k, k_knob = {}, None
    for k in range(1, d + 1):
        acc_k[k] = max(cv_accuracy(('cnn', 'lr'), seed, Hp[:, :, order[:k]], cp, wp, gp, folds).values())
        if k_knob is None and acc_k[k] >= best_full - DELTA:
            k_knob = k
        if k_knob is not None and k >= min(d, max(k_knob + 2, 4)):
            break
    k_knob = k_knob or d
    J = order[:k_knob]
    combos = list(itertools.combinations(range(d), k_knob))
    if len(combos) > 10:
        combos = [combos[i] for i in rng.choice(len(combos), 10, replace=False)]
    acc_random = [max(cv_accuracy(('cnn', 'lr'), seed, Hp[:, :, list(c)], cp, wp, gp, folds).values())
                  for c in combos if set(c) != set(J)] or [float('nan')]
    print(f'  probe set {cells} | acc full {acc_full} | margin-only {acc_margin}\n'
          f'  knobs k={k_knob}: {[names[j] for j in J]} acc={acc_k[k_knob]:.4f} | random-{k_knob} '
          f'{np.mean(acc_random):.4f}+-{np.std(acc_random):.4f} max {np.max(acc_random):.4f} | '
          f'late ratio {late_ratio:.2f}', flush=True)

    # 5. KAN boundary in knob space (+ a random-knob KAN as specificity control)
    kan = fit_kan(seed, Hp[:, :, J], cp, wp)
    alpha = kan.alpha().detach().numpy().astype(np.float64)
    with torch.no_grad():
        kan_train_acc = balanced_accuracy(cp, kan(torch.as_tensor(Hp[:, :, J])).numpy(), wp)
    Jr = np.array(combos[0]) if set(combos[0]) != set(J) else np.array(combos[-1])
    kan_r = fit_kan(seed, Hp[:, :, Jr], cp, wp)
    unc_mu, unc_sd = unc.reshape(len(unc), -1).mean(0), unc.reshape(len(unc), -1).std(0) + 1e-6
    margin_probe = LogisticRegression(max_iter=500).fit((unc.reshape(len(unc), -1) - unc_mu) / unc_sd, cp,
                                                        sample_weight=wp)
    g_margin = lambda m, t: margin_probe.decision_function((np.column_stack([m, np.abs(m - t)]) - unc_mu) / unc_sd)
    state = {}
    for tag, H, m, y in (('validation', Hva, mva, yva), ('test', Hte, mte, yte)):
        phi, grad = kan_state(kan, H[:, :, J])
        dirs, gnorm = directions(grad, seed)
        phi_r, _ = kan_state(kan_r, H[:, :, Jr])
        c_true, w_true = cell_weights(y, m >= thr)
        state[tag] = dict(phi=phi, dirs=dirs, gnorm=gnorm, phi_r=phi_r, correct=c_true, w=w_true, m=m, y=y)
    eps = 0.05 * float(np.median(state['validation']['gnorm']))
    held_out = {}
    for tag, st in state.items():
        c = st['correct']
        held_out[tag] = dict(
            errors=int((c == 0).sum()), FN=int(((c == 0) & (st['y'] == 1)).sum()),
            knob_bal_acc=balanced_accuracy(c, st['phi'], st['w']),
            random_knob_bal_acc=balanced_accuracy(c, st['phi_r'], st['w']),
            margin_only_bal_acc=balanced_accuracy(c, g_margin(st['m'], thr), st['w']),
            knob_error_auc=float(roc_auc_score(c, st['phi'])) if 0 < c.sum() < len(c) else None,
            margin_error_auc=float(roc_auc_score(c, np.abs(st['m'] - thr))) if 0 < c.sum() < len(c) else None,
            random_knob_error_auc=float(roc_auc_score(c, st['phi_r'])) if 0 < c.sum() < len(c) else None)

    # 6. intervention sweep (paper Section 5): normal vs tangent vs random, both signs of tau
    sweep = []
    scores = {'baseline': (mva, mte)}
    for name in state['test']['dirs']:
        for tau in sorted([-t for t in TAUS] + list(TAUS)):
            new = {tag: st['m'] + margin_shift(tau, st['dirs'][name], st['gnorm'], alpha, eps)
                   for tag, st in state.items()}
            e = evaluate(yva, new['validation'], yte, new['test'])
            dll = log_likelihood(yte, new['test']) - log_likelihood(yte, mte)
            wrong = state['test']['correct'] == 0
            sweep.append(dict(direction=name, tau=tau, val_AP=e['validation']['AP'], test_AP=e['test']['AP'],
                              test_F1=e['test']['F1'], test_recall=e['test']['recall'],
                              test_TP=e['test']['TP'], test_FP=e['test']['FP'], test_FN=e['test']['FN'],
                              dlogp_errors=float(dll[wrong].mean()) if wrong.any() else None,
                              dlogp_anomalies=float(dll[yte == 1].mean()),
                              dlogp_normals=float(dll[yte == 0].mean()),
                              improved_errors=float((dll[wrong] > 0).mean()) if wrong.any() else None))
            scores[f'steer-{name}@{tau:+g}'] = (new['validation'], new['test'])

    # 7. boundary-gated rescoring: move a row across the decision threshold in proportion to the
    #    probability that the knob boundary calls the current decision wrong.
    def gate(st, phi, tau):
        return st['m'] + tau * (1 / (1 + np.exp(phi))) * np.where(st['m'] >= thr, -1.0, 1.0)
    gated = []
    for source in ('phi', 'phi_r'):
        for tau in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            new = {tag: gate(st, st[source], tau) for tag, st in state.items()}
            e = evaluate(yva, new['validation'], yte, new['test'])
            label = 'gate-knob' if source == 'phi' else 'gate-random-knob'
            gated.append(dict(source=label, tau=tau, val_AP=e['validation']['AP'], test_AP=e['test']['AP'],
                              test_F1=e['test']['F1'], test_recall=e['test']['recall'],
                              test_TP=e['test']['TP'], test_FP=e['test']['FP'], test_FN=e['test']['FN']))
            scores[f'{label}@{tau:g}'] = (new['validation'], new['test'])

    # 8. boundary-guided retraining: rows the boundary places on the 'incorrect' side get more weight,
    #     then XGBoost is refitted (the intervention goes through the model's own training path).
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    u = {'reweight-knob': 1 / (1 + np.exp(kan_state(kan, Htr[:, :, J])[0])),
         'reweight-random-knob': 1 / (1 + np.exp(kan_state(kan_r, Htr[:, :, Jr])[0])),
         'reweight-margin': 1 / (1 + np.exp(g_margin(mtr, thr_oof)))}
    reweighted = []
    for label, ui in u.items():
        for lam in (1.0, 3.0, 10.0):
            m2 = xgb_model(seed, pos_weight).fit(Xtr, ytr, sample_weight=1 + lam * ui,
                                                 eval_set=[(Xva, yva)], verbose=False)
            sv, stt = logit(m2.predict_proba(Xva)[:, 1]), logit(m2.predict_proba(Xte)[:, 1])
            e = evaluate(yva, sv, yte, stt)
            reweighted.append(dict(source=label, lam=lam, val_AP=e['validation']['AP'], test_AP=e['test']['AP'],
                                   test_F1=e['test']['F1'], test_recall=e['test']['recall'],
                                   test_TP=e['test']['TP'], test_FP=e['test']['FP'], test_FN=e['test']['FN'],
                                   mean_weight=float(1 + lam * ui.mean())))
            scores[f'{label}@{lam:g}'] = (sv, stt)

    # 9. choose the Knob-XGBoost configuration on validation AP (baseline stays a candidate)
    candidates = {k: v for k, v in scores.items()
                  if k == 'baseline' or k.startswith(('steer-normal', 'gate-knob', 'reweight-knob@'))}
    chosen = max(candidates, key=lambda k: (average_precision_score(yva, candidates[k][0]), k == 'baseline'))
    final = evaluate(yva, *[scores[chosen][0]], yte, scores[chosen][1])
    families = {}
    for fam in ('steer-normal', 'steer-tangent', 'steer-random', 'gate-knob@', 'gate-random-knob',
                'reweight-knob@', 'reweight-random-knob', 'reweight-margin'):
        keys = [k for k in scores if k.startswith(fam)]
        if keys:
            best = max(keys, key=lambda k: average_precision_score(yva, scores[k][0]))
            families[fam.rstrip('@')] = dict(config=best, **evaluate(yva, scores[best][0], yte, scores[best][1]))
    alerts = base['test']['TP'] + base['test']['FP']
    missed = (yte == 1) & (mte < thr)
    final_alert = scores[chosen][1] >= final['test']['threshold']
    subtle = dict(baseline_missed=int(missed.sum()),
                  missed_recovered_by_knob=int((missed & final_alert).sum()),
                  caught_lost_by_knob=int(((yte == 1) & (mte >= thr) & ~final_alert).sum()),
                  alert_budget=int(alerts),
                  recall_at_budget_baseline=budget_recall(yte, mte, alerts),
                  recall_at_budget_knob=budget_recall(yte, scores[chosen][1], alerts))
    print(f'  chosen on validation: {chosen}  ->  {brief(final)}\n  subtle: {subtle}', flush=True)

    out = dict(dataset=dataset, variant=variant, seed=seed, trees=int(trees), layers=int(L), features=int(d),
               feature_names=names, baseline=base, probe_set=cells, oof_threshold=thr_oof,
               boundary_cv_balanced_accuracy=dict(full=acc_full, margin_only=acc_margin),
               knobs=dict(k=int(k_knob), indices=[int(j) for j in J], names=[names[j] for j in J],
                          importance={names[j]: float(s[j]) for j in order}, acc_top_k=acc_k,
                          acc_random_mean=float(np.mean(acc_random)), acc_random_sd=float(np.std(acc_random)),
                          acc_random_max=float(np.max(acc_random)), random_subsets=len(acc_random),
                          late_ratio=late_ratio, fold_top2=[[names[j] for j in f] for f in fold_top],
                          delta_map=delta.tolist()),
               kan=dict(layer_attention=alpha.tolist(), train_balanced_accuracy=kan_train_acc,
                        held_out=held_out, eps=eps, random_knob_names=[names[j] for j in Jr]),
               intervention_sweep=sweep, gated=gated, reweighted=reweighted, chosen=chosen, knob_xgb=final, family_best_on_validation=families,
               subtle_anomalies=subtle)
    if save:
        folder = OUT / dataset; folder.mkdir(parents=True, exist_ok=True)
        (folder / f'knob_xgb__{variant}__seed{seed}.json').write_text(json.dumps(out, indent=1) + '\n')
        np.savez_compressed(folder / f'knob_xgb__{variant}__seed{seed}.npz', y_test=yte.astype(np.int8),
                            baseline=mte.astype(np.float64), knob=scores[chosen][1].astype(np.float64),
                            phi=state['test']['phi'].astype(np.float32),
                            threshold_baseline=thr, threshold_knob=final['test']['threshold'])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['sapienza', 'sd-stock', 'sd-crypto'])
    p.add_argument('--variant', default=None)
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    args = p.parse_args()
    variant = args.variant or {'sapienza': 'raw', 'sd-stock': 'causal', 'sd-crypto': 'causal'}[args.dataset]
    splits = {s: load(args.dataset, s) for s in ('train', 'validation', 'test')}
    for seed in map(int, args.seeds.split(',')):
        run(args.dataset, variant, seed, splits)


if __name__ == '__main__':
    main()
