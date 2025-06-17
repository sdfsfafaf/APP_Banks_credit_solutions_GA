# Импорт необходимых библиотек для веб-приложения, обработки данных и визуализации
from flask import Flask, session, render_template, request, redirect, url_for, send_file
from flask_session import Session
import pandas as pd
import numpy as np
import random
import os
import time
import logging
import utils as EGAUtils
import constants as EConstants
import plotly.graph_objects as go
import plotly.io as pio
from multiprocessing import Pool

# Инициализация Flask-приложения
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Секретный ключ для сессий
app.config['SESSION_TYPE'] = 'filesystem'  # Тип хранения сессий
Session(app)  # Активация сессий

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Параметры по умолчанию для генетического алгоритма и банковских расчетов
DEFAULT_PARAMS = {
    'pop_size': 30,  # Размер популяции (уменьшен для оптимизации)
    'generation_size': 30,  # Количество поколений (уменьшено для оптимизации)
    'p_xover': 0.8,  # Вероятность кроссовера
    'p_mutation': 0.006,  # Вероятность мутации
    'reproduction_ratio': 0.194,  # Коэффициент воспроизводства
    'bank_required_reserve_ratio': 0.2,  # Коэффициент резерва банка
    'financial_institutions_deposit': 5000000,  # Объем депозитов
    'rD': 0.009,  # Ставка по депозитам
    'elite_count': 2,  # Количество элитных хромосом
    'bank_predetermined_institutional_cost': 0.0025  # Институциональные затраты
}

# Создание директории для загруженных файлов
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

# Главная страница с перенаправлением на ввод данных
@app.route('/')
def index():
    return redirect(url_for('input'))

# Обработка ввода данных и параметров
@app.route('/input', methods=['GET', 'POST'])
def input():
    session.permanent = True
    loan_types = None
    filter_stats = {}
    clients_table = None
    error = None

    # Обработка загрузки CSV-файла
    if request.method == 'POST':
        logging.info("Получен POST-запрос для загрузки файла")
        # Проверка наличия файла
        if 'file' not in request.files:
            error = "Файл не был отправлен. Пожалуйста, выберите файл."
            logging.error(error)
            return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

        file = request.files['file']  # Загруженный файл
        # Проверка, выбран ли файл
        if not file or file.filename == '':
            error = "Файл не выбран или пуст. Пожалуйста, выберите корректный файл."
            logging.error(error)
            return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

        # Сохранение файла
        filepath = os.path.join('uploaded_files', 'uploaded_data.csv')  # Путь для сохранения файла
        file.save(filepath)
        session['data_file'] = filepath  # Сохранение пути в сессии
        session.modified = True
        logging.info(f"Файл сохранен по пути: {filepath}")

        try:
            # Загрузка и обработка данных
            data = pd.read_csv(filepath, names=EConstants.get_cols_customer(), skiprows=1)  # Чтение CSV
            total_records = len(data)  # Общее количество записей
            logging.info(f"Загружено {total_records} записей")

            # Фильтрация данных
            data['Credit Limit'] = pd.to_numeric(data['Credit Limit'], errors='coerce')
            data['Loan Size'] = pd.to_numeric(data['Loan Size'], errors='coerce')
            data['Loan Age'] = pd.to_numeric(data['Loan Age'], errors='coerce')
            valid_ratings = {'AAA', 'AA', 'A', 'BBB', 'BB'}  # Допустимые рейтинги
            valid_types = ['Mortgage', 'Personal', 'Auto']  # Допустимые типы кредитов

            filtered_data = data[
                (data['Loan Type'].isin(valid_types)) &
                (data['Credit Rating'].isin(valid_ratings)) &
                (~data['Loan Size'].isna()) &
                (~data['Loan Age'].isna()) &
                (~data['Credit Limit'].isna()) &
                (
                    (data['Loan Type'].isin(['Personal', 'Auto']) & (data['Loan Age'] <= 10)) |
                    ((data['Loan Type'] == 'Mortgage') & (data['Loan Age'] > 10) & (data['Loan Age'] <= 20))
                ) &
                (data['Loan Size'] > 0)
            ].reset_index(drop=True)  # Отфильтрованные данные

            # Подсчет статистики фильтрации
            filter_stats = {
                'total': total_records,  # Всего записей
                'filtered': len(filtered_data),  # После фильтрации
                'invalid_type': len(data[~data['Loan Type'].isin(valid_types)]),  # Неверный тип
                'invalid_rating': len(data[~data['Credit Rating'].isin(valid_ratings)]),  # Неверный рейтинг
                'missing_values': len(data[data[['Loan Size', 'Loan Age', 'Credit Limit']].isna().any(axis=1)]),  # Пропуски
                'invalid_age': len(data[~(
                    (data['Loan Type'].isin(['Personal', 'Auto']) & (data['Loan Age'] <= 10)) |
                    ((data['Loan Type'] == 'Mortgage') & (data['Loan Age'] > 10) & (data['Loan Age'] <= 20))
                )]),  # Неверный возраст кредита
                'zero_loan': len(data[data['Loan Size'] <= 0])  # Нулевой или отрицательный размер
            }
            session['filter_stats'] = filter_stats
            session.modified = True
            logging.info(f"Статистика фильтрации: {filter_stats}")

            # Проверка, остались ли данные после фильтрации
            if filtered_data.empty:
                error = "Нет данных после фильтрации. Проверьте формат CSV или данные."
                logging.warning(error)
                return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

            # Сохранение отфильтрованных данных
            filtered_filepath = os.path.join('uploaded_files', 'filtered_data.csv')
            filtered_data.to_csv(filtered_filepath, index=False)
            session['filtered_data_file'] = filtered_filepath
            session['clients_df'] = filtered_data.to_json()
            session.modified = True
            logging.info(f"Сессия обновлена: filtered_data_file={filtered_filepath}, clients_df установлен")

            loan_types = filtered_data['Loan Type'].value_counts().sort_index().to_dict()
            clients_table = filtered_data.to_html(classes='table table-scroll', index=False)

            # Сохранение параметров из формы
            for param, default in DEFAULT_PARAMS.items():
                session[param] = float(request.form.get(param, default))
            session.modified = True
            logging.info(f"Параметры сохранены в сессии: {session}")

        except Exception as e:
            error = f"Ошибка при обработке файла: {str(e)}"
            logging.error(error)
            return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

    return render_template('input.html', loan_types=loan_types, defaults=DEFAULT_PARAMS,
                           filter_stats=filter_stats, clients_table=clients_table, error=error)

# Кэш для хранения значений фитнес-функции
fitness_cache = {}

# Функция для параллельного вычисления фитнеса
def calc_fitness_for_chromo(args):
    # Распаковка аргументов
    df_indices, chromo, rD, bank_required_reserve_ratio, financial_institutions_deposit, bank_predetermined_institutional_cost, data = args
    chromo_key = tuple(chromo)
    # Проверка кэша
    if chromo_key in fitness_cache:
        return fitness_cache[chromo_key]
    selected_indices = df_indices[np.where(chromo)[0]]  # Индексы выбранных клиентов
    df_subset = data.loc[selected_indices]  # Подмножество данных
    # Вычисление фитнеса
    fitness = EGAUtils.calc_fitness(
        df_customers=df_subset,
        bank_required_reserve_ratio=bank_required_reserve_ratio,
        financial_institutions_deposit=financial_institutions_deposit,
        rD=rD,
        bank_predetermined_institutional_cost=bank_predetermined_institutional_cost
    )
    fitness_cache[chromo_key] = fitness  # Сохранение в кэш
    return fitness

# Запуск генетического алгоритма
@app.route('/execute', methods=['GET', 'POST'])
def execute():
    global fitness_cache
    fitness_cache = {}  # Сброс кэша
    logging.info(f"Session contents: {session}")
    # Проверка наличия данных
    if 'filtered_data_file' not in session or 'clients_df' not in session:
        error = "Данные отсутствуют. Пожалуйста, загрузите CSV-файл."
        logging.warning(error)
        return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

    start_time = time.time()  # Начало отсчета времени
    logging.info("Starting genetic algorithm execution")

    try:
        # Загрузка отфильтрованных данных
        filtered_filepath = session['filtered_data_file']  # Путь к данным
        data = pd.read_csv(filtered_filepath)  # Чтение CSV
        num_of_customers = len(data)  # Количество клиентов
        df_indices = data.index.values  # Индексы данных

        # Получение параметров
        params = {key: session[key] for key in DEFAULT_PARAMS.keys()}
        pop_size = int(params['pop_size'])  # Размер популяции
        generation_size = int(params['generation_size'])  # Количество поколений
        p_xover = params['p_xover']  # Вероятность кроссовера
        p_mutation = params['p_mutation']  # Вероятность мутации
        reproduction_ratio = params['reproduction_ratio']  # Коэффициент воспроизводства
        bank_required_reserve_ratio = params['bank_required_reserve_ratio']  # Коэффициент резерва
        financial_institutions_deposit = params['financial_institutions_deposit']  # Депозиты
        rD = params['rD']  # Ставка по депозитам
        elite_count = int(params['elite_count'])  # Количество элитных хромосом
        bank_predetermined_institutional_cost = params['bank_predetermined_institutional_cost']  # Институциональные затраты

        # Инициализация популяции
        chromos = EGAUtils.init_population_with_customers(
            data, num_of_customers, pop_size, bank_required_reserve_ratio, financial_institutions_deposit
        )  # Начальная популяция
        if chromos.size == 0:
            error = "Не удалось создать начальную популяцию. Попробуйте изменить параметры."
            logging.error(error)
            return render_template('output.html', error=error)

        # Вычисление начального фитнеса
        with Pool(processes=4) as pool:
            fitness_args = [(df_indices, chromos[i, :], rD, bank_required_reserve_ratio, financial_institutions_deposit, bank_predetermined_institutional_cost, data) for i in range(chromos.shape[0])]
            chromos_fitness_vector = np.array(pool.map(calc_fitness_for_chromo, fitness_args))  # Вектор фитнеса

        best_fitness_per_generation = []  # Лучший фитнес по поколениям
        avg_fitness_per_generation = []  # Средний фитнес по поколениям

        # Эволюционный цикл
        for gen in range(generation_size):
            logging.info(f"Generation {gen + 1}/{generation_size}")
            # Селекция родителей
            selected_indices = EGAUtils.roulette_wheel_selection(chromos, chromos_fitness_vector)  # Индексы выбранных хромосом
            number_of_worst_chromo_to_be_deleted = 0  # Количество хромосом для удаления

            # Обработка пар родителей
            for i in range(0, len(selected_indices), 2):
                if i + 1 < len(selected_indices):
                    parent1 = chromos[selected_indices[i], :]  # Первый родитель
                    parent2 = chromos[selected_indices[i + 1], :]  # Второй родитель

                    # Воспроизводство
                    if random.random() < reproduction_ratio:
                        chromos = np.vstack((chromos, parent1.copy()))
                        number_of_worst_chromo_to_be_deleted += 1

                    # Кроссовер и мутация
                    rand_xover = random.random()  # Случайное значение для кроссовера
                    rand_mutation = random.random()  # Случайное значение для мутации
                    is_xovered = False
                    is_mutated = False
                    xover_ch1 = np.zeros(num_of_customers, dtype=bool)  # Первый потомок от кроссовера
                    xover_ch2 = np.zeros(num_of_customers, dtype=bool)  # Второй потомок от кроссовера
                    mut_ch1 = np.zeros(num_of_customers, dtype=bool)  # Первый потомок от мутации
                    mut_ch2 = np.zeros(num_of_customers, dtype=bool)  # Второй потомок от мутации

                    if rand_xover <= p_xover:
                        xover_ch1, xover_ch2 = EGAUtils.xover(parent1, parent2)
                        is_xovered = True
                    if rand_mutation <= p_mutation:
                        mut_ch1, mut_ch2 = EGAUtils.mutate(parent1, parent2)
                        is_mutated = True

                    # Добавление потомков от кроссовера
                    if is_xovered:
                        if EGAUtils.is_GAMCC_satisfied_vec(xover_ch1, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, xover_ch1))
                            number_of_worst_chromo_to_be_deleted += 1
                        if EGAUtils.is_GAMCC_satisfied_vec(xover_ch2, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, xover_ch2))
                            number_of_worst_chromo_to_be_deleted += 1

                    # Добавление потомков от мутации
                    if is_mutated:
                        if EGAUtils.is_GAMCC_satisfied_vec(mut_ch1, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, mut_ch1))
                            number_of_worst_chromo_to_be_deleted += 1
                        if EGAUtils.is_GAMCC_satisfied_vec(mut_ch2, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, mut_ch2))
                            number_of_worst_chromo_to_be_deleted += 1

            # Удаление худших хромосом
            chromos = EGAUtils.delete_chromo_based_on_bad_fit(chromos, chromos_fitness_vector, number_of_worst_chromo_to_be_deleted)

            # Пересчет фитнеса
            with Pool(processes=4) as pool:
                fitness_args = [(df_indices, chromos[i, :], rD, bank_required_reserve_ratio, financial_institutions_deposit, bank_predetermined_institutional_cost, data) for i in range(chromos.shape[0])]
                chromos_fitness_vector = np.array(pool.map(calc_fitness_for_chromo, fitness_args))

            # Сохранение элитных хромосом
            elite_indices = np.argsort(chromos_fitness_vector)[-elite_count:]  # Индексы лучших хромосом
            elite_chromos = chromos[elite_indices, :]  # Элитные хромосомы
            chromos = np.vstack((elite_chromos, chromos[:-elite_count, :]))  # Обновленная популяция

            best_fitness_per_generation.append(np.max(chromos_fitness_vector))  # Лучший фитнес
            avg_fitness_per_generation.append(np.mean(chromos_fitness_vector))  # Средний фитнес

        # Выбор лучшего решения
        max_fit_index = np.argmax(chromos_fitness_vector)  # Индекс лучшей хромосомы
        best_solution = chromos[max_fit_index, :]  # Лучшая хромосома
        best_df = data.iloc[np.where(best_solution)[0], :]  # Данные выбранных клиентов

        if best_df.empty:
            error = "Алгоритм не нашел подходящих клиентов. Попробуйте изменить параметры."
            logging.warning(error)
            return render_template('output.html', error=error)

        execution_time = time.time() - start_time  # Время выполнения
        logging.info(f"Execution time: {execution_time:.2f} seconds")

        # Подсчет статистики
        count_of_accepted_customer = best_df.shape[0]  # Количество принятых клиентов
        loan_type_counts = best_df['Loan Type'].value_counts()  # Подсчет типов кредитов
        credit_rating_counts = best_df['Credit Rating'].value_counts()  # Подсчет рейтингов

        loan_type_percentages = {k: (v / count_of_accepted_customer * 100) if count_of_accepted_customer > 0 else 0 for k, v in loan_type_counts.items()}  # Процентное распределение типов
        credit_rating_percentages = {k: (v / count_of_accepted_customer * 100) if count_of_accepted_customer > 0 else 0 for k, v in credit_rating_counts.items()}  # Процентное распределение рейтингов

        # Сохранение результатов
        session['results'] = {
            'best_fitness': float(chromos_fitness_vector[max_fit_index]),  # Лучший фитнес
            'total_loan_size': float(best_df['Loan Size'].sum()),  # Общий размер кредитов
            'expected_loss': float(EGAUtils.calc_sum_of_landa(best_df)),  # Ожидаемые потери
            'accepted_customers': count_of_accepted_customer,  # Количество клиентов
            'execution_time': execution_time,  # Время выполнения
            'best_df': best_df.to_json(),  # Данные в JSON
            'filter_stats': session.get('filter_stats', {})  # Статистика фильтрации
        }
        session['graph_data'] = {
            'loan_type_percentages': loan_type_percentages,  # Процент типов кредитов
            'credit_rating_percentages': credit_rating_percentages,  # Процент рейтингов
            'best_fitness_per_generation': [float(f) for f in best_fitness_per_generation],  # Лучший фитнес по поколениям
            'avg_fitness_per_generation': [float(f) for f in avg_fitness_per_generation]  # Средний фитнес по поколениям
        }
        session.modified = True
        logging.info(f"Results saved to session: {session['results']}")

        # Сохранение результатов в файл
        result_row = {
            'M%': loan_type_percentages.get('Mortgage', 0) / 100,  # Доля ипотек
            'P%': loan_type_percentages.get('Personal', 0) / 100,  # Доля персональных кредитов
            'LA%': loan_type_percentages.get('Auto', 0) / 100,  # Доля автокредитов
            'D': financial_institutions_deposit,  # Депозиты
            'POP_SIZE': pop_size,  # Размер популяции
            'AAA%': credit_rating_percentages.get('AAA', 0) / 100,  # Доля рейтинга AAA
            'AA%': credit_rating_percentages.get('AA', 0) / 100,  # Доля рейтинга AA
            'A%': credit_rating_percentages.get('A', 0) / 100,  # Доля рейтинга A
            'BBB%': credit_rating_percentages.get('BBB', 0) / 100,  # Доля рейтинга BBB
            'BB%': credit_rating_percentages.get('BB', 0) / 100,  # Доля рейтинга BB
            'ACCEPTED_CUSTOMERS': count_of_accepted_customer,  # Количество клиентов
            'GENERATION_SIZE': generation_size  # Количество поколений
        }
        results_file = 'results.txt'  # Файл для результатов
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            df_results = pd.read_csv(results_file, names=EConstants.get_cols_result())  # Чтение существующих результатов
        else:
            df_results = pd.DataFrame(columns=EConstants.get_cols_result())  # Новый DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([result_row])], ignore_index=True)  # Добавление строки
        df_results.to_csv(results_file, index=False, header=False)  # Сохранение в файл

        return redirect(url_for('output'))

    except Exception as e:
        error = f"Ошибка во время выполнения алгоритма: {str(e)}"
        logging.error(error)
        return render_template('output.html', error=error)

# Отображение результатов
@app.route('/output')
def output():
    # Проверка наличия результатов
    if 'results' not in session or 'graph_data' not in session:
        error = "Результаты не найдены. Пожалуйста, запустите алгоритм."
        logging.warning(error)
        return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

    # Загрузка данных для визуализации
    best_df = pd.read_json(session['results']['best_df'])  # Лучшие клиенты
    loan_type_percentages = session['graph_data']['loan_type_percentages']  # Процент типов кредитов
    credit_rating_percentages = session['graph_data']['credit_rating_percentages']  # Процент рейтингов
    best_fitness_per_generation = session['graph_data']['best_fitness_per_generation']  # Лучший фитнес
    avg_fitness_per_generation = session['graph_data']['avg_fitness_per_generation']  # Средний фитнес

    # Создание графиков
    # График распределения типов кредитов
    fig_types = go.Figure(data=[
        go.Bar(
            x=list(loan_type_percentages.keys()),
            y=list(loan_type_percentages.values()),
            marker_color=['#FF6384', '#36A2EB', '#FFCE56']  # Цвета столбцов
        )
    ])
    fig_types.update_layout(
        title='Распределение типов кредитов (%)',
        xaxis_title='Тип кредита',
        yaxis_title='Процент'
    )

    # График распределения рейтингов
    fig_ratings = go.Figure(data=[
        go.Bar(
            x=list(credit_rating_percentages.keys()),
            y=list(credit_rating_percentages.values()),
            marker_color=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
        )
    ])
    fig_ratings.update_layout(
        title='Распределение кредитных рейтингов (%)',
        xaxis_title='Рейтинг',
        yaxis_title='Процент'
    )

    # Гистограмма возраста кредитов
    fig_age = go.Figure(data=[
        go.Histogram(
            x=best_df['Loan Age'],
            nbinsx=10,  # Количество интервалов
            marker_color='#36A2EB'
        )
    ])
    fig_age.update_layout(
        title='Гистограмма возраста кредитов',
        xaxis_title='Возраст (годы)',
        yaxis_title='Частота'
    )

    # Гистограмма размера кредитов
    fig_size = go.Figure(data=[
        go.Histogram(
            x=best_df['Loan Size'],
            nbinsx=10,
            marker_color='#FFCE56'
        )
    ])
    fig_size.update_layout(
        title='Гистограмма размера кредитов',
        xaxis_title='Размер кредита',
        yaxis_title='Частота'
    )

    # Гистограмма кредитных лимитов
    fig_limit = go.Figure(data=[
        go.Histogram(
            x=best_df['Credit Limit'],
            nbinsx=10,
            marker_color='#4BC0C0'
        )
    ])
    fig_limit.update_layout(
        title='Гистограмма кредитных лимитов',
        xaxis_title='Кредитный лимит',
        yaxis_title='Частота'
    )

    # График сходимости фитнеса
    fig_fitness = go.Figure()
    fig_fitness.add_trace(go.Scatter(
        y=best_fitness_per_generation,
        mode='lines',
        name='Лучший фитнес',
        line=dict(color='#36A2EB')
    ))
    fig_fitness.add_trace(go.Scatter(
        y=avg_fitness_per_generation,
        mode='lines',
        name='Средний фитнес',
        line=dict(color='#FF6384')
    ))
    fig_fitness.update_layout(
        title='Сходимость фитнеса',
        xaxis_title='Поколение',
        yaxis_title='Фитнес'
    )

    # Подготовка графиков для отображения
    graphs = {
        'types': pio.to_html(fig_types, full_html=False),
        'ratings': pio.to_html(fig_ratings, full_html=False),
        'age': pio.to_html(fig_age, full_html=False),
        'size': pio.to_html(fig_size, full_html=False),
        'limit': pio.to_html(fig_limit, full_html=False),
        'fitness': pio.to_html(fig_fitness, full_html=False)
    }

    return render_template('output.html', results=session['results'], graphs=graphs,
                           clients_table=best_df.to_html(classes='table table-scroll', index=False))

# Скачивание результатов
@app.route('/download_results')
def download_results():
    # Проверка наличия результатов
    if 'results' not in session:
        error = "Результаты не найдены. Пожалуйста, запустите алгоритм."
        logging.warning(error)
        return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)
    results = session['results']  # Результаты из сессии
    df = pd.DataFrame([results])  # DataFrame с результатами
    df.to_csv('results_export.csv', index=False)  # Сохранение в CSV
    return send_file('results_export.csv', as_attachment=True)  # Отправка файла

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)