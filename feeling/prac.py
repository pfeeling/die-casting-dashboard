import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('data/train_v1.csv')

df.value_counts()
df.info()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False





import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    if col not in ['id', 'registration_time']:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

X = df_encoded.drop(columns=['passorfail', 'registration_time'])
y = df_encoded['passorfail']

# 🔹 XGBoost DMatrix 사용
model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# ✅ SHAP TreeExplainer 사용 (xgboost 전용)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 🔹 SHAP DataFrame 생성
shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df['mold_code'] = df['mold_code'].values

# 🔹 보고 싶은 변수 5개
selected_features = ['biscuit_thickness', 'lower_mold_temp2', 'upper_mold_temp1',
                     'upper_mold_temp2', 'cast_pressure']

mold_code_top5_positive = []

for mold, group in shap_df.groupby('mold_code'):
    # 양수 기여도만 평균
    mean_shap = group[selected_features].apply(lambda x: x[x > 0].mean())
    
    # 높은 순 정렬 후 top 5
    top5 = mean_shap.sort_values(ascending=False).dropna()[:5]
    
    # 결과 저장
    row = {'mold_code': mold}
    for var, val in top5.items():
        row[var] = round(val, 5)
    
    mold_code_top5_positive.append(row)

# 최종 DataFrame
top5_positive_df = pd.DataFrame(mold_code_top5_positive)

# 확인
print(top5_positive_df.head())




# ----------------test, train 데이터 유사성 확인--------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

# 데이터 로드
train_v1 = pd.read_csv('data/train_v1.csv')

# 불필요 컬럼 제거
drop_cols = ['id', 'registration_time']
train_v1 = train_v1.drop(columns=drop_cols)

# 범주형 변수 인코딩
for col in ['working', 'tryshot_signal', 'heating_furnace', 'mold_code']:
    le = LabelEncoder()
    train_v1[col] = le.fit_transform(train_v1[col])

# 타겟/피처 분리
X = train_v1.drop(columns=['passorfail'])
y = train_v1['passorfail']

# 데이터 분할 (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost 모델 정의 (불균형 대응)
xgb_model = XGBClassifier(
    random_state=42,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='logloss',
    use_label_encoder=False,
)

# 모델 학습
xgb_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = xgb_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.4f}")
print(classification_report(y_test, y_pred))

from sklearn.inspection import permutation_importance

# permutation importance 계산
result = permutation_importance(
    xgb_model, X_test, y_test,
    n_repeats=10,   # 반복 횟수(높을수록 안정적)
    random_state=42,
    n_jobs=-1       # 모든 코어 사용
)


# 결과 정리 (중요도 내림차순)
importances = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(importances)

# ----------------------------------------
import matplotlib.pyplot as plt

# 상위 10개 변수만 추출
topn = 10
top_features = importances.head(topn)

plt.figure(figsize=(8, 5))
plt.barh(top_features['feature'][::-1], top_features['importance_mean'][::-1], 
         xerr=top_features['importance_std'][::-1])
plt.xlabel("Permutation Importance (Mean)")
plt.title(f"Top {topn} Feature Importances (Permutation)")
plt.tight_layout()
plt.show()

# -------------------------------
import shap

# 1️ TreeExplainer 객체 생성
explainer = shap.TreeExplainer(xgb_model)

# 2️ SHAP 값 계산 (X_test 대상)
shap_values = explainer.shap_values(X_test)

# 3️ summary_plot (전체 변수별 영향도)
shap.summary_plot(shap_values, X_test, plot_type="bar")     # 막대그래프(평균 절대값)
shap.summary_plot(shap_values, X_test)                      # 점 구름 plot(분포와 영향 동시)



from xgboost import to_graphviz
dot = to_graphviz(xgb_model, num_trees=0, rankdir='LR', with_stats=True)
dot.render('xgb_tree0_with_stats.dot')




















import pandas as pd


xgb_model.get_booster().dump_model('tree_dump.txt', with_stats=True)

def get_leaf_conditions(tree_txt_path):
    with open(tree_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    path_stack = []
    leaf_info = {}
    for line in lines:
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        if ':' in content and 'leaf=' not in content:
            # 분기 노드
            cond = content.split('[')[1].split(']')[0]
            path_stack = path_stack[:indent//2] + [cond]
        elif 'leaf=' in content:
            leaf_idx = int(content.split(':')[0])
            cond_path = ' & '.join(path_stack)
            leaf_info[leaf_idx] = cond_path
    return leaf_info

leaf_condition_map = get_leaf_conditions('tree_dump.txt')


# 트리 0번 리프 index 추출
leaf_nodes = xgb_model.apply(X_test)[:, 0]
result_df = pd.DataFrame({'leaf': leaf_nodes, 'label': y_test.values})

# 리프별 양품/불량 count
leaf_stats = result_df.groupby('leaf')['label'].value_counts().unstack(fill_value=0)
leaf_stats.columns = ['양품(0)', '불량(1)']

# 전체 샘플수 및 불량비율 추가
leaf_stats['전체 샘플수'] = leaf_stats['양품(0)'] + leaf_stats['불량(1)']
leaf_stats['불량비율'] = leaf_stats['불량(1)'] / leaf_stats['전체 샘플수']

# 조건 경로 연결
leaf_stats['조건경로'] = leaf_stats.index.map(lambda idx: leaf_condition_map.get(idx, ''))

# 결과 확인 (상위 10개)
print(leaf_stats[['양품(0)', '불량(1)', '전체 샘플수', '불량비율', '조건경로']].head(10))