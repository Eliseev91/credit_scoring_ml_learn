from prepare_data import load_and_prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Подготовленные данные
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Обучаем Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Предсказание
y_pred_rf = rf_model.predict(X_test)

# Оценка модели
print("📊 Random Forest — Отчёт по классификации:")
print(classification_report(y_test, y_pred_rf))
print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred_rf))