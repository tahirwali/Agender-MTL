"""
The multi-task network for age and gender, written in NumPy with hand-written
backpropagation.

    shared trunk : d -> 64 -> 32   (ReLU, dropout 0.3 after each layer)
    gender head  : 32 -> 1         (sigmoid)
    age head     : [32 + detached gender probability] = 33 -> 16 -> C   (ReLU, softmax)

Loss 2 * L_gender + L_age, both class-weighted by softened inverse frequency (power 0.5).
For a table without gender, gender_weight=0 trains the age head alone (the gender head then learns nothing).
AdamW, learning rate 1e-3 -> 1e-5 by a cosine schedule over 150 epochs, weight decay 5e-4,
batch 256; the weights with the lowest validation loss are kept.
"""
import numpy as np


def class_weights(labels, n_classes, power=0.5):
    """Weight of each class proportional to (1 / count)^power, scaled so the average sample weight is 1."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = np.where(counts > 0, 1.0 / np.maximum(counts, 1) ** power, 0.0)
    return weights / (weights[labels].mean())


class MultiTaskNet:
    def __init__(self, n_inputs, n_age, trunk=(64, 32), age_hidden=16, dropout=0.3, seed=0, gender_weight=2.0):
        self.gender_weight = gender_weight
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout
        shapes = {"W1": (n_inputs, trunk[0]), "W2": (trunk[0], trunk[1]), "Wg": (trunk[1], 1),
                  "Wa1": (trunk[1] + 1, age_hidden), "Wa2": (age_hidden, n_age)}
        self.params = {}
        for name, (fan_in, fan_out) in shapes.items():
            self.params[name] = self.rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            self.params["b" + name[1:]] = np.zeros(fan_out)

    # ---- forward ----
    def _forward(self, X, train):
        p = self.params
        cache = {"X": X}
        h = X
        for layer in ("1", "2"):
            z = h @ p["W" + layer] + p["b" + layer]
            h = np.maximum(z, 0)
            mask = None
            if train and self.dropout > 0:
                mask = (self.rng.random(h.shape) >= self.dropout) / (1 - self.dropout)
                h = h * mask
            cache["z" + layer], cache["h" + layer], cache["mask" + layer] = z, h, mask
        p_gender = 1 / (1 + np.exp(-(h @ p["Wg"] + p["bg"])[:, 0]))
        # the gender probability enters the age head but no gradient flows back through it
        age_input = np.column_stack([h, p_gender])
        za1 = age_input @ p["Wa1"] + p["ba1"]
        ha1 = np.maximum(za1, 0)
        logits = ha1 @ p["Wa2"] + p["ba2"]
        logits = logits - logits.max(axis=1, keepdims=True)
        p_age = np.exp(logits)
        p_age /= p_age.sum(axis=1, keepdims=True)
        cache.update(p_gender=p_gender, age_input=age_input, za1=za1, ha1=ha1, p_age=p_age)
        return p_gender, p_age, cache

    def predict(self, X):
        p_gender, p_age, _ = self._forward(np.asarray(X, dtype=float), train=False)
        return p_gender, p_age

    # ---- loss and gradients ----
    def loss(self, p_gender, p_age, y_gender, y_age, w_gender, w_age):
        eps = 1e-12
        sw_g, sw_a = w_gender[y_gender], w_age[y_age]
        l_gender = -np.mean(sw_g * (y_gender * np.log(p_gender + eps) + (1 - y_gender) * np.log(1 - p_gender + eps)))
        l_age = -np.mean(sw_a * np.log(p_age[np.arange(len(y_age)), y_age] + eps))
        return self.gender_weight * l_gender + l_age

    def _gradients(self, cache, y_gender, y_age, w_gender, w_age):
        p = self.params
        n = len(y_gender)
        grads = {}
        # age head: weighted cross-entropy through softmax
        d_logits = cache["p_age"].copy()
        d_logits[np.arange(n), y_age] -= 1
        d_logits *= (w_age[y_age] / n)[:, None]
        grads["Wa2"] = cache["ha1"].T @ d_logits
        grads["ba2"] = d_logits.sum(axis=0)
        d_za1 = (d_logits @ p["Wa2"].T) * (cache["za1"] > 0)
        grads["Wa1"] = cache["age_input"].T @ d_za1
        grads["ba1"] = d_za1.sum(axis=0)
        upstream = (d_za1 @ p["Wa1"].T)[:, :-1]      # drop the detached gender column
        # gender head: weighted binary cross-entropy through sigmoid, times the gender weight
        d_zg = (self.gender_weight * w_gender[y_gender] * (cache["p_gender"] - y_gender) / n)[:, None]
        grads["Wg"] = cache["h2"].T @ d_zg
        grads["bg"] = d_zg.sum(axis=0)
        upstream = upstream + d_zg @ p["Wg"].T
        # trunk
        for layer, below in (("2", cache["h1"]), ("1", cache["X"])):
            if cache["mask" + layer] is not None:
                upstream = upstream * cache["mask" + layer]
            d_z = upstream * (cache["z" + layer] > 0)
            grads["W" + layer] = below.T @ d_z
            grads["b" + layer] = d_z.sum(axis=0)
            upstream = d_z @ p["W" + layer].T
        return grads

    # ---- training ----
    def fit(self, X, y_gender, y_age, X_val, y_gender_val, y_age_val, n_age, epochs=150, batch=256,
            lr=1e-3, lr_end=1e-5, weight_decay=5e-4, betas=(0.9, 0.999)):
        X = np.asarray(X, dtype=float)
        X_val = np.asarray(X_val, dtype=float)
        w_gender = class_weights(y_gender, 2)
        w_age = class_weights(y_age, n_age)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        step = 0
        best = (np.inf, None)
        self.history = []
        for epoch in range(epochs):
            rate = lr_end + 0.5 * (lr - lr_end) * (1 + np.cos(np.pi * epoch / epochs))
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), batch):
                rows = order[start:start + batch]
                _, _, cache = self._forward(X[rows], train=True)
                grads = self._gradients(cache, y_gender[rows], y_age[rows], w_gender, w_age)
                step += 1
                for name, g in grads.items():
                    m[name] = betas[0] * m[name] + (1 - betas[0]) * g
                    v[name] = betas[1] * v[name] + (1 - betas[1]) * g * g
                    m_hat = m[name] / (1 - betas[0] ** step)
                    v_hat = v[name] / (1 - betas[1] ** step)
                    self.params[name] -= rate * weight_decay * self.params[name]     # decoupled decay
                    self.params[name] -= rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            train_loss = self.loss(*self.predict(X), y_gender, y_age, w_gender, w_age)
            val_loss = self.loss(*self.predict(X_val), y_gender_val, y_age_val, w_gender, w_age)
            self.history.append((train_loss, val_loss))
            if val_loss < best[0]:
                best = (val_loss, {k: w.copy() for k, w in self.params.items()})
        self.params = best[1]
        self.best_epoch = int(np.argmin([h[1] for h in self.history]))
        return self
