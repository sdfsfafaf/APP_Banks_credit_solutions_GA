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

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Параметры по умолчанию
DEFAULT_PARAMS = {
    'pop_size': 30,  # Уменьшено для оптимизации
    'generation_size': 30,  # Уменьшено для оптимизации
    'p_xover': 0.8,
    'p_mutation': 0.01,  # Увеличено для ускорения поиска
    'reproduction_ratio': 0.194,
    'bank_required_reserve_ratio': 0.2,
    'financial_institutions_deposit': 5000000,
    'rD': 0.009,
    'elite_count': 2,
    'bank_predetermined_institutional_cost': 0.005
}

# Создание директории для загрузки файлов
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

@app.route('/')
def index():
    return redirect(url_for('input'))

@app.route('/input', methods=['GET', 'POST'])
def input():
    session.permanent = True
    loan_types = None
    filter_stats = {}
    clients_table = None
    error = None

    if request.method == 'POST':
        logging.info("Получен POST-запрос для загрузки файла")
        if 'file' not in request.files:
            error = "Файл не был отправлен. Пожалуйста, выберите файл."
            logging.error(error)
            return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

        file = request.files['file']
        if not file or file.filename == '':
            error = "Файл не выбран или пуст. Пожалуйста, выберите корректный файл."
            logging.error(error)
            return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

        filepath = os.path.join('uploaded_files', 'uploaded_data.csv')
        file.save(filepath)
        session['data_file'] = filepath
        session.modified = True
        logging.info(f"Файл сохранен по пути: {filepath}")

        try:
            # Загрузка данных
            data = pd.read_csv(filepath, names=EConstants.get_cols_customer(), skiprows=1)
            total_records = len(data)
            logging.info(f"Загружено {total_records} записей")

            # Фильтрация с подсчетом причин
            data['Credit Limit'] = pd.to_numeric(data['Credit Limit'], errors='coerce')
            data['Loan Size'] = pd.to_numeric(data['Loan Size'], errors='coerce')
            data['Loan Age'] = pd.to_numeric(data['Loan Age'], errors='coerce')
            valid_ratings = {'AAA', 'AA', 'A', 'BBB', 'BB'}
            valid_types = ['Mortgage', 'Personal', 'Auto']

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
            ].reset_index(drop=True)

            # Статистика фильтрации
            filter_stats = {
                'total': total_records,
                'filtered': len(filtered_data),
                'invalid_type': len(data[~data['Loan Type'].isin(valid_types)]),
                'invalid_rating': len(data[~data['Credit Rating'].isin(valid_ratings)]),
                'missing_values': len(data[data[['Loan Size', 'Loan Age', 'Credit Limit']].isna().any(axis=1)]),
                'invalid_age': len(data[~(
                    (data['Loan Type'].isin(['Personal', 'Auto']) & (data['Loan Age'] <= 10)) |
                    ((data['Loan Type'] == 'Mortgage') & (data['Loan Age'] > 10) & (data['Loan Age'] <= 20))
                )]),
                'zero_loan': len(data[data['Loan Size'] <= 0])
            }
            session['filter_stats'] = filter_stats  # Сохранение статистики в сессии
            session.modified = True
            logging.info(f"Статистика фильтрации: {filter_stats}")

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

            # Сохранение параметров
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

fitness_cache = {}  # Кэш для фитнес-функции

def calc_fitness_for_chromo(args):
    df_indices, chromo, rD, bank_required_reserve_ratio, financial_institutions_deposit, bank_predetermined_institutional_cost, data = args
    chromo_key = tuple(chromo)
    if chromo_key in fitness_cache:
        return fitness_cache[chromo_key]
    selected_indices = df_indices[np.where(chromo)[0]]
    df_subset = data.loc[selected_indices]
    fitness = EGAUtils.calc_fitness(
        df_customers=df_subset,
        bank_required_reserve_ratio=bank_required_reserve_ratio,
        financial_institutions_deposit=financial_institutions_deposit,
        rD=rD,
        bank_predetermined_institutional_cost=bank_predetermined_institutional_cost
    )
    fitness_cache[chromo_key] = fitness
    return fitness

@app.route('/execute', methods=['GET', 'POST'])
def execute():
    global fitness_cache
    fitness_cache = {}  # Сброс кэша перед выполнением
    logging.info(f"Session contents: {session}")
    if 'filtered_data_file' not in session or 'clients_df' not in session:
        error = "Данные отсутствуют. Пожалуйста, загрузите CSV-файл."
        logging.warning(error)
        return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

    start_time = time.time()
    logging.info("Starting genetic algorithm execution")

    try:
        # Загрузка данных
        filtered_filepath = session['filtered_data_file']
        data = pd.read_csv(filtered_filepath)
        num_of_customers = len(data)
        df_indices = data.index.values

        # Параметры из сессии
        params = {key: session[key] for key in DEFAULT_PARAMS.keys()}
        pop_size = int(params['pop_size'])
        generation_size = int(params['generation_size'])
        p_xover = params['p_xover']
        p_mutation = params['p_mutation']
        reproduction_ratio = params['reproduction_ratio']
        bank_required_reserve_ratio = params['bank_required_reserve_ratio']
        financial_institutions_deposit = params['financial_institutions_deposit']
        rD = params['rD']
        elite_count = int(params['elite_count'])
        bank_predetermined_institutional_cost = params['bank_predetermined_institutional_cost']

        # Инициализация популяции
        chromos = EGAUtils.init_population_with_customers(
            data, num_of_customers, pop_size, bank_required_reserve_ratio, financial_institutions_deposit
        )
        if chromos.size == 0:
            error = "Не удалось создать начальную популяцию. Попробуйте изменить параметры."
            logging.error(error)
            return render_template('output.html', error=error)

        # Параллельное вычисление фитнеса
        with Pool(processes=4) as pool:
            fitness_args = [(df_indices, chromos[i, :], rD, bank_required_reserve_ratio, financial_institutions_deposit, bank_predetermined_institutional_cost, data) for i in range(chromos.shape[0])]
            chromos_fitness_vector = np.array(pool.map(calc_fitness_for_chromo, fitness_args))

        best_fitness_per_generation = []
        avg_fitness_per_generation = []

        # Эволюционный цикл
        for gen in range(generation_size):
            logging.info(f"Generation {gen + 1}/{generation_size}")
            selected_indices = EGAUtils.tournament_selection(chromos, chromos_fitness_vector, tournament_size=3)
            number_of_worst_chromo_to_be_deleted = 0

            for i in range(0, len(selected_indices), 2):
                if i + 1 < len(selected_indices):
                    parent1 = chromos[selected_indices[i], :]
                    parent2 = chromos[selected_indices[i + 1], :]

                    if random.random() < reproduction_ratio:
                        chromos = np.vstack((chromos, parent1.copy()))
                        number_of_worst_chromo_to_be_deleted += 1

                    rand_xover = random.random()
                    rand_mutation = random.random()
                    is_xovered = False
                    is_mutated = False
                    xover_ch1 = np.zeros(num_of_customers, dtype=bool)
                    xover_ch2 = np.zeros(num_of_customers, dtype=bool)
                    mut_ch1 = np.zeros(num_of_customers, dtype=bool)
                    mut_ch2 = np.zeros(num_of_customers, dtype=bool)

                    if rand_xover <= p_xover:
                        xover_ch1, xover_ch2 = EGAUtils.xover(parent1, parent2)
                        is_xovered = True
                    if rand_mutation <= p_mutation:
                        mut_ch1, mut_ch2 = EGAUtils.mutate(parent1, parent2)
                        is_mutated = True

                    if is_xovered:
                        if EGAUtils.is_GAMCC_satisfied_vec(xover_ch1, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, xover_ch1))
                            number_of_worst_chromo_to_be_deleted += 1
                        if EGAUtils.is_GAMCC_satisfied_vec(xover_ch2, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, xover_ch2))
                            number_of_worst_chromo_to_be_deleted += 1

                    if is_mutated:
                        if EGAUtils.is_GAMCC_satisfied_vec(mut_ch1, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, mut_ch1))
                            number_of_worst_chromo_to_be_deleted += 1
                        if EGAUtils.is_GAMCC_satisfied_vec(mut_ch2, data, bank_required_reserve_ratio, financial_institutions_deposit):
                            chromos = np.vstack((chromos, mut_ch2))
                            number_of_worst_chromo_to_be_deleted += 1

            chromos = EGAUtils.delete_chromo_based_on_bad_fit(chromos, chromos_fitness_vector, number_of_worst_chromo_to_be_deleted)

            with Pool(processes=4) as pool:
                fitness_args = [(df_indices, chromos[i, :], rD, bank_required_reserve_ratio, financial_institutions_deposit, bank_predetermined_institutional_cost, data) for i in range(chromos.shape[0])]
                chromos_fitness_vector = np.array(pool.map(calc_fitness_for_chromo, fitness_args))

            elite_indices = np.argsort(chromos_fitness_vector)[-elite_count:]
            elite_chromos = chromos[elite_indices, :]
            chromos = np.vstack((elite_chromos, chromos[:-elite_count, :]))

            best_fitness_per_generation.append(np.max(chromos_fitness_vector))
            avg_fitness_per_generation.append(np.mean(chromos_fitness_vector))

        max_fit_index = np.argmax(chromos_fitness_vector)
        best_solution = chromos[max_fit_index, :]
        best_df = data.iloc[np.where(best_solution)[0], :]

        if best_df.empty:
            error = "Алгоритм не нашел подходящих клиентов. Попробуйте изменить параметры."
            logging.warning(error)
            return render_template('output.html', error=error)

        execution_time = time.time() - start_time
        logging.info(f"Execution time: {execution_time:.2f} seconds")

        # Статистика для результатов
        count_of_accepted_customer = best_df.shape[0]
        loan_type_counts = best_df['Loan Type'].value_counts()
        credit_rating_counts = best_df['Credit Rating'].value_counts()

        loan_type_percentages = {k: (v / count_of_accepted_customer * 100) if count_of_accepted_customer > 0 else 0 for k, v in loan_type_counts.items()}
        credit_rating_percentages = {k: (v / count_of_accepted_customer * 100) if count_of_accepted_customer > 0 else 0 for k, v in credit_rating_counts.items()}

        # Сохранение результатов
        session['results'] = {
            'best_fitness': float(chromos_fitness_vector[max_fit_index]),
            'total_loan_size': float(best_df['Loan Size'].sum()),
            'expected_loss': float(EGAUtils.calc_sum_of_landa(best_df)),
            'accepted_customers': count_of_accepted_customer,
            'execution_time': execution_time,
            'best_df': best_df.to_json(),
            'filter_stats': session.get('filter_stats', {})
        }
        session['graph_data'] = {
            'loan_type_percentages': loan_type_percentages,
            'credit_rating_percentages': credit_rating_percentages,
            'best_fitness_per_generation': [float(f) for f in best_fitness_per_generation],
            'avg_fitness_per_generation': [float(f) for f in avg_fitness_per_generation]
        }
        session.modified = True
        logging.info(f"Results saved to session: {session['results']}")

        # Сохранение результатов в файл
        result_row = {
            'M%': loan_type_percentages.get('Mortgage', 0) / 100,
            'P%': loan_type_percentages.get('Personal', 0) / 100,
            'LA%': loan_type_percentages.get('Auto', 0) / 100,
            'D': financial_institutions_deposit,
            'POP_SIZE': pop_size,
            'AAA%': credit_rating_percentages.get('AAA', 0) / 100,
            'AA%': credit_rating_percentages.get('AA', 0) / 100,
            'A%': credit_rating_percentages.get('A', 0) / 100,
            'BBB%': credit_rating_percentages.get('BBB', 0) / 100,
            'BB%': credit_rating_percentages.get('BB', 0) / 100,
            'ACCEPTED_CUSTOMERS': count_of_accepted_customer,
            'GENERATION_SIZE': generation_size
        }
        results_file = 'results.txt'
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            df_results = pd.read_csv(results_file, names=EConstants.get_cols_result())
        else:
            df_results = pd.DataFrame(columns=EConstants.get_cols_result())
        df_results = pd.concat([df_results, pd.DataFrame([result_row])], ignore_index=True)
        df_results.to_csv(results_file, index=False, header=False)

        return redirect(url_for('output'))

    except Exception as e:
        error = f"Ошибка во время выполнения алгоритма: {str(e)}"
        logging.error(error)
        return render_template('output.html', error=error)

@app.route('/output')
def output():
    if 'results' not in session or 'graph_data' not in session:
        error = "Результаты не найдены. Пожалуйста, запустите алгоритм."
        logging.warning(error)
        return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)

    # Загрузка данных
    best_df = pd.read_json(session['results']['best_df'])
    loan_type_percentages = session['graph_data']['loan_type_percentages']
    credit_rating_percentages = session['graph_data']['credit_rating_percentages']
    best_fitness_per_generation = session['graph_data']['best_fitness_per_generation']
    avg_fitness_per_generation = session['graph_data']['avg_fitness_per_generation']

    # Графики Plotly
    fig_types = go.Figure(data=[
        go.Bar(
            x=list(loan_type_percentages.keys()),
            y=list(loan_type_percentages.values()),
            marker_color=['#FF6384', '#36A2EB', '#FFCE56']
        )
    ])
    fig_types.update_layout(
        title='Распределение типов кредитов (%)',
        xaxis_title='Тип кредита',
        yaxis_title='Процент'
    )

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

    fig_age = go.Figure(data=[
        go.Histogram(
            x=best_df['Loan Age'],
            nbinsx=10,
            marker_color='#36A2EB'
        )
    ])
    fig_age.update_layout(
        title='Гистограмма возраста кредитов',
        xaxis_title='Возраст (годы)',
        yaxis_title='Частота'
    )

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

@app.route('/download_results')
def download_results():
    if 'results' not in session:
        error = "Результаты не найдены. Пожалуйста, запустите алгоритм."
        logging.warning(error)
        return render_template('input.html', error=error, defaults=DEFAULT_PARAMS)
    results = session['results']
    df = pd.DataFrame([results])
    df.to_csv('results_export.csv', index=False)
    return send_file('results_export.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)