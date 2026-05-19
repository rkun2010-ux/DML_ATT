import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy import stats
import matplotlib.pyplot as plt


def estimate_ey1(X, D, Y, ps_method='l', poly_degree=4):
    if ps_method == 'l':
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)

    elif ps_method == 'p':
        model = make_pipeline(
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            LogisticRegression(penalty=None, C=1.0, solver='lbfgs', max_iter=2000)
        )

    elif ps_method == 'n':
        model = MLPClassifier(hidden_layer_sizes=(12, 12),
                              activation='relu',
                              solver='adam',
                              max_iter=1000,
                              random_state=42)
    else:
        raise ValueError("Unknown method")
    model.fit(X, D)
    e_hat = model.predict_proba(X)[:, 1]
    e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)
    ipw_scores = (D * Y) / e_hat
    ey1_ipw = np.mean(ipw_scores)
    #############################################################################
    if ps_method == 'l':
        or_model = LinearRegression()
    elif ps_method == 'p':
        or_model = make_pipeline(
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            Ridge(alpha=0.0)
        )
    elif ps_method == 'n':
        or_model = MLPRegressor(hidden_layer_sizes=(12, 12), activation='relu',
                                solver='adam', max_iter=1000, random_state=42)
    mask1 = (D == 1)
    or_model.fit(X[mask1], Y[mask1])
    mu1_hat = or_model.predict(X)

    add_score = (1 - D / e_hat) * mu1_hat * np.sqrt(len(D))
    mean_add_score = np.mean(add_score)

    e = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.5 * X[:, 1])))
    mu1 = 1.0 + 2 * X[:, 0] + 2 * X[:, 1] + 2
    add_score_star = (1 - D / e_hat) * mu1 * np.sqrt(len(D))
    mean_add_score_star = np.mean(add_score_star)

    return mean_add_score_star, mean_add_score, ey1_ipw


num_simulations = 200
sample_size = 3000


def generate_data(n):
    X = np.random.uniform(-1, 1, (n, 2))
    logits = 0.5 * X[:, 0] - 0.5 * X[:, 1]
    true_probs = 1 / (1 + np.exp(-logits))
    D = np.random.binomial(1, true_probs)

    Y_noise = np.random.normal(0, 1, n)
    Y = 1.0 + 2 * X[:, 0] + 2 * X[:, 1] + 2.0 * D + Y_noise
    return X, D, Y


results_ipw = []
results_dr = []
ses_ipw = []
ses_dr = []
add_score_means = []
np.random.seed(42)

for i in range(num_simulations):
    X, D, Y = generate_data(n=sample_size)
    ey1_ipw, add_score, IPW = estimate_ey1(X, D, Y, ps_method='n')

    results_ipw.append(ey1_ipw)
    add_score_means.append(add_score)
    results_dr.append(IPW)

mean_add = np.mean(add_score_means)
emp_sd_add = np.std(add_score_means)

print("=" * 60)
print(f"{'Metric':<20} | {'IPW':<15}  ")
print("-" * 60)
print(f"{'True E[Y(1)]':<20} | {3.0000:<15.4f}  ")
print(f"{'Mean Est E[Y(1)]':<20}| {np.mean(results_dr):<15.4f}| {np.mean(results_ipw):<15.4f}| {np.std(results_ipw):<15.4f} | {mean_add:<15.4f} | {emp_sd_add:<15.4f} ")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

stats.probplot((add_score_means - np.mean(add_score_means)) / np.std(add_score_means), dist="norm", plot=ax1)
ax1.set_title("")
ax1.set_xlabel(fr"Theoretical quantiles (standardized $B_n[\hat{{h}}_n]$)")
ax1.set_ylabel(r"Sample Quantiles")

stats.probplot((results_ipw - np.mean(results_ipw)) / np.std(results_ipw), dist="norm", plot=ax2)
ax2.set_title("")
ax2.set_xlabel(fr"Theoretical quantiles (standardized $B_n[h^*]$)")
ax2.set_ylabel("")

for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

plt.tight_layout()
plt.show()
