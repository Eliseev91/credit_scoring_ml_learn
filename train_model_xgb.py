from prepare_data import load_and_prepare_data
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Загружаем подготовленные данные
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Обучаем XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=1.0,  # можем изменить позже для дисбаланса
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Предсказания
y_pred = xgb_model.predict(X_test)

# Оценка качества
print("📊 XGBoost — Отчёт по классификации:")
print(classification_report(y_test, y_pred))
print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred))