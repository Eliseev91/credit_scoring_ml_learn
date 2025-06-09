from prepare_data import load_and_prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Загружаем данные
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Обучение логистической регрессии
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)


# Оценка результатов
print("📊 Матрица ошибок (confusion matrix):")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Отчёт по классификации:")
print(classification_report(y_test, y_pred))

print("🎯 Accuracy (точность):", accuracy_score(y_test, y_pred))