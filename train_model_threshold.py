from prepare_data import load_and_prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Данные
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Модель
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Вероятности
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Тестируем разные пороги
for threshold in [0.5, 0.45, 0.4, 0.35]:
    print(f"\n📌 Threshold = {threshold}")
    y_pred = (y_pred_proba >= threshold).astype(int)
    print(classification_report(y_test, y_pred))
    print("Матрица ошибок:\n", confusion_matrix(y_test, y_pred))