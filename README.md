# credit_scoring_ml_learn
Учебный проект по машинному обучению: построение модели для оценки кредитного риска клиентов.  A learning project in machine learning: building a model to assess client credit risk.

## 📌 Цель проекта

Разработать модель, которая по данным о клиенте предсказывает, является ли он потенциально надёжным или рискованным заёмщиком.

## 📁 Структура проекта
<pre lang="text"><code>
credit_scoring_ml_learn/
│
├── data/ # Данные (не выкладываются на GitHub) 
├── prepare_data.py # Загрузка и подготовка данных
├── eda.py # Первичный анализ данных 
├── eda_visuals.py # Визуализация признаков и зависимостей
├── train_model.py # Обучение модели логистической регрессии 
├── requirements.txt # Зависимости
└── README.md # Этот файл </code></pre>

## 🔧 Используемые библиотеки

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 🚀 Как запустить

- Клонируй проект:
git clone https://github.com/ТВОЙ_ЛОГИН/credit_scoring_ml_learn.git
cd credit_scoring_ml_learn

- Установи зависимости:
pip install -r requirements.txt
- Скачай датасет и помести его в папку data/ с именем credit_data.csv

- Запускай файлы по порядку:
prepare_data.py
eda.py / eda_visuals.py
train_model.py

⚠️ Важно
Файл с данными (credit_data.csv) не выкладывается на GitHub и должен быть добавлен вручную.

