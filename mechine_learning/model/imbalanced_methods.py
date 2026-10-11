
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from process_data.dm05_get_data import prepare_data

# ==================== 1. 数据准备 ====================
# 必须和 rf_methods.py 完全一致，保证可比
# dm03挑选的
#selected_features = ['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预']
selected_features = ['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预', 'Smoking', 'CCI', 'bleeding', 'surgerytime', '袖切类型1（1-简单袖切，2=扩大袖切，3=袖+部分叶）', 'FVC实测值', 'FVC实/预', 'FEV1/FVC实测值', 'DLCO实测值']
# # dm04挑选的
# selected_features = ['APTT', 'FEV1实/预', 'surgerytime', 'FEV1/FVC预计值']
# selected_features = ['APTT', 'FEV1实/预', 'surgerytime', 'FEV1/FVC预计值', 'Dimer', 'bleeding', 'FEV1/FVC实/预', '切除肺叶_5', 'BMI', 'DLCO实/预', '胸膜黏连', 'CCI', 'DLCO预计值', 'FVC实/预', 'fibrinogen', 'FEV1/FVC实测值', 'DLCO实测值', '白蛋白', 'LYM']

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
    file_path="./data/肺科多模态数据-处理后.xlsx",
    sheet_name="Sheet1",
    selected_features=selected_features,
    use_lobe_features=True,
    test_size=0.2,
    random_state=42
)

X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y_train = np.asarray(y_train).astype(int).ravel()
y_test = np.asarray(y_test).astype(int).ravel()

print("训练集标签分布：")
print(pd.Series(y_train).value_counts())
print("测试集标签分布：")
print(pd.Series(y_test).value_counts())

# ==================== 2. 依赖 ====================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)
from imblearn.ensemble import (
    BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ==================== 3. 统一评估函数 ====================
results = {}

def evaluate(name, y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        auc, pr_auc = np.nan, np.nan
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 62}")
    print(f"模型：{name}")
    print(f"{'=' * 62}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print("混淆矩阵：")
    print(cm)

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'MCC': mcc, 'ROC-AUC': auc, 'PR-AUC': pr_auc
    }


# ==================== 4. BalancedRandomForest ====================
print("\n>>> 方法 1：BalancedRandomForest")
brf = BalancedRandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
brf.fit(X_train, y_train)
proba = brf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("BalancedRandomForest", y_test, pred, proba)


# ==================== 5. EasyEnsemble ====================
print("\n>>> 方法 2：EasyEnsemble")
ee = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)
ee.fit(X_train, y_train)
proba = ee.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("EasyEnsemble", y_test, pred, proba)


# ==================== 6. RUSBoost ====================
print("\n>>> 方法 3：RUSBoost")
rus = RUSBoostClassifier(
    n_estimators=200,
    random_state=42
)
rus.fit(X_train, y_train)
proba = rus.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("RUSBoost", y_test, pred, proba)


# ==================== 7. 异常检测：多数类为正常 ====================
X_normal = X_train[y_train == 0]
print(f"\n异常检测训练集（多数类）样本数：{len(X_normal)}")

# ---- 7.1 IsolationForest ----
print("\n>>> 方法 4：IsolationForest")
iso = IsolationForest(
    n_estimators=300,
    contamination='auto',
    random_state=42,
    n_jobs=-1
)
iso.fit(X_normal)
score_iso = -iso.score_samples(X_test)
thr_iso = np.percentile(-iso.score_samples(X_normal), 95)
pred_iso = (score_iso >= thr_iso).astype(int)
evaluate("IsolationForest", y_test, pred_iso, score_iso)


# ---- 7.2 OneClassSVM ----
print("\n>>> 方法 5：OneClassSVM")
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm.fit(X_normal)
score_oc = -ocsvm.decision_function(X_test)
thr_oc = np.percentile(-ocsvm.decision_function(X_normal), 95)
pred_oc = (score_oc >= thr_oc).astype(int)
evaluate("OneClassSVM", y_test, pred_oc, score_oc)


# ---- 7.3 Autoencoder (PyTorch) ----
print("\n>>> 方法 6：Autoencoder")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    mu = X_normal.mean(axis=0, keepdims=True)
    sigma = X_normal.std(axis=0, keepdims=True) + 1e-8
    X_normal_std = (X_normal - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    class AE(nn.Module):
        def __init__(self, in_dim, hidden=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, hidden), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, in_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = AE(in_dim=X_normal_std.shape[1]).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_normal_std, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    ae.train()
    for epoch in range(100):
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = ae(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()

    def ae_score(X):
        ae.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(device)
            out = ae(xb)
            return ((out - xb) ** 2).mean(dim=1).cpu().numpy()

    score_ae_train = ae_score(X_normal_std)
    score_ae_test = ae_score(X_test_std)
    thr_ae = np.percentile(score_ae_train, 95)
    pred_ae = (score_ae_test >= thr_ae).astype(int)
    evaluate("Autoencoder", y_test, pred_ae, score_ae_test)

except ImportError:
    print("未安装 PyTorch，跳过 Autoencoder。")


# ---- 7.4 VAE (PyTorch) ----
print("\n>>> 方法 7：VAE")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    mu_n = X_normal.mean(axis=0, keepdims=True)
    sigma_n = X_normal.std(axis=0, keepdims=True) + 1e-8
    X_normal_std = (X_normal - mu_n) / sigma_n
    X_test_std = (X_test - mu_n) / sigma_n

    class VAE(nn.Module):
        def __init__(self, in_dim, hidden=8):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.fc_mu = nn.Linear(16, hidden)
            self.fc_logvar = nn.Linear(16, hidden)
            self.dec = nn.Sequential(
                nn.Linear(hidden, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, in_dim)
            )

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.dec(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    def vae_loss(recon_x, x, mu, logvar, beta=1.0):
        recon = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon + beta * kl, recon, kl

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(in_dim=X_normal_std.shape[1], hidden=8).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    train_ds = TensorDataset(torch.tensor(X_normal_std, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    vae.train()
    n_epochs = 150
    for epoch in range(n_epochs):
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu_z, logvar_z = vae(batch)
            loss, _, _ = vae_loss(recon, batch, mu_z, logvar_z, beta=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 30 == 0:
            print(f"  VAE epoch {epoch+1}/{n_epochs}, loss={total_loss/len(train_ds):.4f}")

    def vae_score(X):
        vae.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(device)
            recon, _, _ = vae(xb)
            return ((recon - xb) ** 2).mean(dim=1).cpu().numpy()

    score_vae_train = vae_score(X_normal_std)
    score_vae_test = vae_score(X_test_std)
    thr_vae = np.percentile(score_vae_train, 95)
    pred_vae = (score_vae_test >= thr_vae).astype(int)
    evaluate("VAE", y_test, pred_vae, score_vae_test)

except ImportError:
    print("未安装 PyTorch，跳过 VAE。")


# ==================== 8. 汇总对比 ====================
print("\n\n" + "=" * 62)
print("非 RF 方法对比汇总（按 PR-AUC 降序）")
print("=" * 62)
summary = pd.DataFrame(results).T
summary = summary.sort_values('PR-AUC', ascending=False)
print(summary.round(4))

# summary.to_csv("./imbalanced_methods_results.csv", encoding="utf-8-sig")
# print("\n结果已保存到 ./imbalanced_methods_results.csv")

