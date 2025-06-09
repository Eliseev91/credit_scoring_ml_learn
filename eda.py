import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Настройка отображения датасета
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

# Настройка отображения графиков
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# Загрузка данных
df = pd.read_csv('data/credit_data.csv')


# Первые строки
print("🔹 Первые строки:")
print(df.head())

# Размер данных
print("\n🔹 Размер датасета (строки, колонки):", df.shape)

# Типы данных и пропуски
print("\n🔹 Информация о колонках:")
print(df.info())

# Статистика по числовым колонкам
print("\n🔹 Описательная статистика:")
print(df.describe())

# Проверка пропущенных значений
print("\n🔹 Количество пропущенных значений:")
print(df.isnull().sum())
