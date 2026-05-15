import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


# ---------------------------------------------------------
# 1. DGP
# ---------------------------------------------------------
def generate_data(n, p=2):
    X = np.random.normal(0, 1, size=(n, p))
    f_true_x = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    pi_true = 1 / (1 + np.exp(-f_true_x))
    D = np.random.binomial(1, pi_true)
    mu1_true = 1.5 + 0.8 * X[:, 0] + 0.5 * X[:, 1]
    return X, D, pi_true, mu1_true


# ---------------------------------------------------------
# 2. 
# ---------------------------------------------------------
def run_simulation(method='ml', n_simulations=200, n_samples=3000):
    bn_h_hat_list = []
    bn_h_star_list = []
    for _ in tqdm(range(n_simulations)):
        X, D, pi_true, mu1_true = generate_data(n=n_samples)

        if method == 'ml':
            clf = RandomForestClassifier(n_estimators=50, max_depth=5)
            reg = RandomForestRegressor(n_estimators=50, max_depth=5)

        elif method == 'linear':
            clf = LogisticRegression(penalty=None, max_iter=1000)
            reg = Ridge(alpha=1.0)

        elif method == 'spline':
            spline = SplineTransformer(n_knots=5, degree=3)
            clf = make_pipeline(spline, LogisticRegression(penalty='l2', C=1.0, max_iter=2000))
            reg = make_pipeline(spline, Ridge(alpha=1.0))

        clf.fit(X, D)
        pi_hat = np.clip(clf.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)

        Y1_observed = mu1_true + np.random.normal(0, 0.5, size=len(mu1_true))
        reg.fit(X[D == 1], Y1_observed[D == 1])
        mu1_hat = reg.predict(X)

        h_hat = - mu1_hat / pi_hat
        h_star = - mu1_true / pi_true

        score_part = pi_hat - D

        bn_h_hat_list.append(np.mean(score_part * h_hat) * np.sqrt(len(D)))
        bn_h_star_list.append(np.mean(score_part * h_star) * np.sqrt(len(D)))

    return bn_h_hat_list, bn_h_star_list


chosen_method = 'ml'
res_hat, res_star = run_simulation(method=chosen_method)

def standardize(data):
    print(np.std(data))
    print(np.mean(data))
    return (data - np.mean(data)) / np.std(data)


z_h_hat = standardize(res_hat)
z_h_star = standardize(res_star)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

stats.probplot(z_h_hat, dist="norm", plot=ax1)
ax1.set_title("")
ax1.set_xlabel(fr"Theoretical Quantiles ($B_n[\hat{{h}}_n]$)")
ax1.set_ylabel(r"Sample Quantiles")

stats.probplot(z_h_star, dist="norm", plot=ax2)
ax2.set_title("")
ax2.set_xlabel(fr"Theoretical Quantiles ($B_n[h^*]$)")
ax2.set_ylabel("")

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

plt.tight_layout()
plt.show()
