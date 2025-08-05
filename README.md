# credit_scoring_ml_learn
Учебный проект по машинному обучению: построение модели для оценки кредитного риска клиентов.  A learning project in machine learning: building a model to assess client credit risk.

## 📌 Цель проекта

Разработать модель, которая по данным о клиенте предсказывает, является ли он потенциально надёжным или рискованным заёмщиком.

## 📁 Структура проекта
<pre lang="text"><code>
credit_scoring_ml_learn/
│
├── data/                    # Папка с данными (не выкладывается на GitHub) 
├── notebooks/               # Jupyter notebooks с исследованиями
├── prepare_data.py          # Загрузка и подготовка данных
├── eda.py                   # Первичный анализ данных 
├── eda_visuals.py           # Визуализация признаков и зависимостей
├── train_model.py           # Обучение базовой модели логистической регрессии
├── train_model_rf.py        # Обучение модели Random Forest
├── train_model_xgb.py       # Обучение модели XGBoost
├── train_model_xgb_smote.py # Обучение XGBoost с применением SMOTE
├── train_model_threshold.py # Эксперименты с порогами классификации
├── requirements.txt         # Зависимости проекта
└── README.md               # Этот файл </code></pre>

## 🔧 Используемые библиотеки

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- imbalanced-learn (для SMOTE)

## 🚀 Как запустить

- Клонируй проект:
git clone https://github.com/ТВОЙ_ЛОГИН/credit_scoring_ml_learn.git
cd credit_scoring_ml_learn

- Установи зависимости:
pip install -r requirements.txt
- Скачай датасет и помести его в папку data/ с именем credit_data.csv

- Запускай файлы по порядку:
1. prepare_data.py
2. eda.py / eda_visuals.py
3. train_model.py (или другие модели: train_model_rf.py, train_model_xgb.py, train_model_xgb_smote.py)

⚠️ Важно
Файл с данными (credit_data.csv) не выкладывается на GitHub и должен быть добавлен вручную.
