import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    # Загрузка данных

    df = pd.read_csv('data/credit_data.csv')

    # 1. Отделяем целевую переменную
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk']

    # 2. Кодируем категориальные признаки
    X = pd.get_dummies(X, drop_first=True)

    # 4. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify — чтобы сохранить баланс классов
     )

    # 5. Масштабирование числовых признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train,y_test

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_prepare_data()

    print("✅ Данные успешно загружены и подготовлены!")
    print("Форма X_train:", X_train_scaled.shape)
    print("Форма X_test:", X_test_scaled.shape)
    print("Классы в y_train:", y_train.value_counts(normalize=True))
    print("Первые 5 строк X_train:\n", X_train_scaled[:5])