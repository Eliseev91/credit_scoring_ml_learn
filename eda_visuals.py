import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data/credit_data.csv')

# Настройки графиков
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Распределение целевой переменной
plt.figure()
sns.countplot(x='credit_risk', data=df)
plt.title('Распределение целевой переменной (credit_risk)')
plt.xlabel('Кредитный риск (0 = хороший, 1 = плохой)')
plt.ylabel('Количество клиентов')
plt.show()

# 2. Распределение суммы кредита по классам риска
plt.figure()
sns.histplot(data=df, x='amount', hue='credit_risk', kde=True, bins=30, multiple='stack')
plt.title('Распределение суммы кредита по классам риска')
plt.xlabel('Сумма кредита')
plt.ylabel('Количество клиентов')
plt.show()

# 3. Возраст по классам риска
plt.figure()
sns.boxplot(x='credit_risk', y='age', data=df)
plt.title('Возраст клиентов по классам риска')
plt.xlabel('Кредитный риск')
plt.ylabel('Возраст')
plt.show()

# 4. Корреляционная матрица
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Корреляция между признаками')
plt.show()