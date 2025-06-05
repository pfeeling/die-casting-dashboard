# ──────────────────────────────────────────────────────────────
# 라이브러리 정리
# ──────────────────────────────────────────────────────────────
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

from lightgbm import LGBMClassifier, create_tree_digraph

from imblearn.over_sampling import SMOTE

import shap  # 추가: SHAP 분석용


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 1) 데이터 로드 및 X, y 분리
# ──────────────────────────────────────────────────────────────
# train.csv는 이미 필요한 전처리가 완료된 상태라고 가정합니다.
df = pd.read_csv("train.csv")

# (필요에 따라 df에 추가 전처리가 있다면 여기에 삽입)

# 특성(X)과 레이블(y) 분리
X = df.drop(columns=['id', 'passorfail'])
y = df['passorfail']


# ──────────────────────────────────────────────────────────────
# 2) 범주형 Encoding
# ──────────────────────────────────────────────────────────────
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ──────────────────────────────────────────────────────────────
# 3) StratifiedShuffleSplit 기반 교차검증 & LightGBM 학습
# ──────────────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
model_lgbm = LGBMClassifier(random_state=42)

xgb_fold_f1 = []
idx_list = []
n_iter = 0

print("=== StratifiedShuffleSplit 기반 교차검증 시작 ===")
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for inner_train_idx, valid_idx in sss.split(X_train, y_train):
        X_inner_train, X_valid = X_train.iloc[inner_train_idx], X_train.iloc[valid_idx]
        y_inner_train, y_valid = y_train.iloc[inner_train_idx], y_train.iloc[valid_idx]

        # 정규화: MinMaxScaler
        scaler = MinMaxScaler()
        X_inner_train_scaled = scaler.fit_transform(X_inner_train)
        X_valid_scaled       = scaler.transform(X_valid)
        X_test_scaled        = scaler.transform(X_test)

        # LightGBM 학습
        model_lgbm.fit(X_inner_train_scaled, y_inner_train)

        # 검증, 테스트 예측
        valid_pred = model_lgbm.predict(X_valid_scaled)
        test_pred  = model_lgbm.predict(X_test_scaled)

        valid_f1 = f1_score(y_valid, valid_pred)
        test_f1  = f1_score(y_test,  test_pred)

        n_iter += 1
        xgb_fold_f1.append([n_iter, test_f1])
        idx_list.append([train_idx, test_idx])

        print(f"[Fold {n_iter}] Valid F1: {valid_f1:.4f}  |  Test F1: {test_f1:.4f}")
    print("--------------------------------------------------")

# Test F1 기준으로 상위 1개 분할 정보 추출
xgb_fold_f1.sort(key=lambda x: x[1], reverse=True)
best_iter, best_f1 = xgb_fold_f1[0]
print(f"\n>> 가장 높은 Test F1: Fold {best_iter} | F1 = {best_f1:.4f}")

# 최적 분할의 train_idx
best_split_idx = best_iter - 1
best_train_idx = idx_list[best_split_idx][0]
X_train_best   = X.iloc[best_train_idx]
y_train_best   = y.iloc[best_train_idx]


# ──────────────────────────────────────────────────────────────
# 4) 최적 학습 데이터로 LightGBM 재학습 → 트리 시각화
# ──────────────────────────────────────────────────────────────
# Graphviz(dot) 경로 설정 (환경에 따라 경로 수정 필요)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

model_best = LGBMClassifier(random_state=42)
model_best.fit(X_train_best, y_train_best)

# 첫 번째 트리(tree_index=0) Graphviz 시각화
tree_graph = create_tree_digraph(
    model_best,
    tree_index=0,
    show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count']
)
tree_graph.render(view=True)  # PNG로 렌더링 후 자동으로 열기


# ──────────────────────────────────────────────────────────────
# 8) train_v1.csv 예시: 결측 대체 → 인코딩 → SMOTE → LightGBM 평가
# ──────────────────────────────────────────────────────────────
train_v1 = pd.read_csv("train_v1.csv")

# mold_code 문자열 변환 & 범주형 인코딩
train_v1['mold_code'] = train_v1['mold_code'].astype(str)
cat_cols_v1 = train_v1.select_dtypes(include='object').columns
for col in cat_cols_v1:
    le = LabelEncoder()
    train_v1[col] = le.fit_transform(train_v1[col].astype(str))

# X/y 분리
X_v1 = train_v1.drop(columns=['id', 'passorfail'])
y_v1 = train_v1['passorfail']

# 수치형/범주형 분리
num_cols_v1 = X_v1.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols_v1 = X_v1.select_dtypes(include='object').columns.tolist()

# 결측 대체: 수치형은 평균
#imputer_v1 = SimpleImputer(strategy='mean')
#X_v1[num_cols_v1] = imputer_v1.fit_transform(X_v1[num_cols_v1])

# Train/Validation 분리 (70%/30%), Stratify 적용
X_train_v1, X_val_v1, y_train_v1, y_val_v1 = train_test_split(
    X_v1, y_v1, test_size=0.3, random_state=1449, stratify=y_v1
)

# SMOTE 적용 (훈련 데이터만)
smote = SMOTE(random_state=42)
X_train_res_v1, y_train_res_v1 = smote.fit_resample(X_train_v1, y_train_v1)

# LightGBM 학습
model_v1 = LGBMClassifier(random_state=42)
model_v1.fit(X_train_res_v1, y_train_res_v1)

# 검증 데이터 예측 및 평가
y_pred_v1 = model_v1.predict(X_val_v1)
print(">>> train_v1.csv 기준 LightGBM 평가 지표")
print("  - F1 Score:", f1_score(y_val_v1, y_pred_v1, average='macro'))
print("  - Recall :", recall_score(y_val_v1, y_pred_v1, average='macro'))
print("  - Confusion Matrix:\n", confusion_matrix(y_val_v1, y_pred_v1))
print("  - Classification Report:\n", classification_report(y_val_v1, y_pred_v1))



# ──────────────────────────────────────────────────────────────
# 9) SHAP 분석
# ──────────────────────────────────────────────────────────────
# (1) TreeExplainer 생성 및 SHAP 값 계산
explainer = shap.TreeExplainer(model_v1)
# 훈련 데이터 일부(또는 전체)에 대해서 SHAP 값을 계산할 수 있음. 
# 예시로 X_train_res_v1 전체에 대해 계산:
shap_values = explainer.shap_values(X_train_res_v1)

# (2) SHAP summary plot (피처 중요도 및 분포)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_train_res_v1, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_res_v1, show=False)
plt.title("SHAP Summary Plot (Dot)")
plt.tight_layout()
plt.show()

# (3) SHAP dependence plot 예시 (가장 중요한 피처 몇 개 선택)
# 상위 2개 피처만 예시로 그려봄:
feature_names_sorted = np.array(X_train_res_v1.columns)[np.argsort(np.abs(shap_values).mean(0))[::-1]]
top_feats = feature_names_sorted[:2]

for feat in top_feats:
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(feat, shap_values, X_train_res_v1, show=False)
    plt.title(f"SHAP Dependence Plot: {feat}")
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────
# 9) PDP(평균 영향) / ICE(개별 종속성) 시각화 (예시)
# ──────────────────────────────────────────────────────────────
# X_val_v1 인코딩(범주형 → 숫자) 예시:  
X_val_enc = X_val_v1.copy()
for col in cat_cols_v1:
    # 실제 상황에서는 train에서 사용한 LabelEncoder 객체를 재사용해야 함
    X_val_enc[col] = LabelEncoder().fit_transform(X_val_enc[col].astype(str))

# PDP 대상 피처 리스트 (예시)
pdp_features = [
    "cast_pressure", "tryshot_signal", "upper_mold_temp1", "lower_mold_temp2",
    "lower_mold_temp1", "low_section_speed", "upper_mold_temp2", "sleeve_temperature",
    "registration_time", "mold_code", "high_section_speed", "physical_strength",
    "biscuit_thickness", "molten_temp", "count"
]

print("\n>>> PDP / ICE 시각화 시작")
for feat in pdp_features:
    # PDP (평균 영향)
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            estimator=model_v1,
            X=X_val_enc,
            features=[feat],
            kind="average",
            grid_resolution=50,
            ax=ax
        )
        ax.set_title(f"PDP for '{feat}'")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"--- PDP 에러: '{feat}', 내용: {e}")
        plt.close()

    # ICE (개별 종속성)
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            estimator=model_v1,
            X=X_val_enc,
            features=[feat],
            kind="individual",
            grid_resolution=50,
            ax=ax
        )
        ax.set_title(f"ICE for '{feat}'")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"--- ICE 에러: '{feat}', 내용: {e}")
        plt.close()


# ──────────────────────────────────────────────────────────────
# 10) Permutation Importance 계산 및 시각화 (train_v1 예시)
# ──────────────────────────────────────────────────────────────
perm_result = permutation_importance(
    estimator=model_v1,
    X=X_val_v1,
    y=y_val_v1,
    n_repeats=10,
    random_state=42,
    scoring='f1_macro'
)

perm_imp_df = pd.DataFrame({
    'feature': X_val_v1.columns.tolist(),
    'importance_mean': perm_result.importances_mean,
    'importance_std' : perm_result.importances_std
})
perm_imp_df = perm_imp_df.sort_values(by='importance_mean', ascending=False).reset_index(drop=True)

print("\n>>> Permutation Importance (내림차순 정렬)")
print(perm_imp_df)

# 상위 15개 피처 막대그래프
top_n = 15
plt.figure(figsize=(8, 6))
plt.barh(
    y=perm_imp_df.loc[:top_n-1, 'feature'][::-1],
    width=perm_imp_df.loc[:top_n-1, 'importance_mean'][::-1],
    xerr=perm_imp_df.loc[:top_n-1, 'importance_std'][::-1],
    align='center',
    color='skyblue',
    ecolor='gray',
    capsize=3
)
plt.xlabel("Mean Decrease in f1_macro (Permutation Importance)")
plt.title(f"Top {top_n} Feature Permutation Importances")
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────
# 12) 주요 변수만 사용한 Surrogate Decision Tree (재시각화)
# ──────────────────────────────────────────────────────────────
selected_features = [
    "cast_pressure", "upper_mold_temp1", "lower_mold_temp2", "lower_mold_temp1",
    "low_section_speed", "upper_mold_temp2", "sleeve_temperature",
    "high_section_speed", "physical_strength", "biscuit_thickness"
]

# X_train_best에서 선택된 10개 피처만 추출
X_train_sel = X_train_best[selected_features]

# 이미 생성된 y_surrogate 사용 (model_best.predict 결과)
dt_sel = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_sel.fit(X_train_sel, y_surrogate)

plt.figure(figsize=(50, 12))
plot_tree(
    dt_sel,
    feature_names=selected_features,
    class_names=['0','1'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Surrogate Decision Tree (선택된 10개 피처)")
plt.tight_layout()
plt.show()