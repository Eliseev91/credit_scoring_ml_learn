from prepare_data import load_and_prepare_data
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Подготовка данных
X_train, X_test, y_train, y_test = load_and_prepare_data()

# 2. Применяем SMOTE только к тренировочным данным
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"До SMOTE: {y_train.value_counts().to_dict()}")
print(f"После SMOTE: {y_train_res.value_counts().to_dict()}")

# 3. Обучаем XGBoost на сбалансированных данных
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_res, y_train_res)

# 4. Предсказания
y_pred = xgb_model.predict(X_test)

# 5. Оценка качества
print("📊 XGBoost + SMOTE — Отчёт по классификации:")
print(classification_report(y_test, y_pred))
print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred))