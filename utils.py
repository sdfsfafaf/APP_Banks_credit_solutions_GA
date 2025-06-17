# Импорт библиотек для обработки данных
import pandas as pd
import numpy as np
import random

# Получение процентных ставок для кредитов
def get_loan_interest_rate_vec(loan_types, loan_ages):
    """Векторная функция для получения процентных ставок."""
    r_L = np.zeros_like(loan_types, dtype=float)  # Массив процентных ставок
    mask_mortgage = loan_types == 'Mortgage'  # Маска для ипотек
    mask_personal = loan_types == 'Personal'  # Маска для персональных кредитов
    mask_auto = loan_types == 'Auto'  # Маска для автокредитов

    # Назначение ставок для ипотек
    mask_age_mortgage = (10 < loan_ages) & (loan_ages <= 20)  # Возраст 10–20 лет
    r_L[mask_mortgage & mask_age_mortgage] = np.random.uniform(0.021, 0.028, size=np.sum(mask_mortgage & mask_age_mortgage))

    # Назначение ставок для персональных кредитов
    mask_age_p3 = loan_ages <= 3  # Возраст до 3 лет
    mask_age_p5 = (loan_ages > 3) & (loan_ages <= 5)  # Возраст 3–5 лет
    mask_age_p10 = (loan_ages > 5) & (loan_ages <= 10)  # Возраст 5–10 лет
    r_L[mask_personal & mask_age_p3] = np.random.uniform(0.0599, 0.0601, size=np.sum(mask_personal & mask_age_p3))
    r_L[mask_personal & mask_age_p5] = np.random.uniform(0.0601, 0.0604, size=np.sum(mask_personal & mask_age_p5))
    r_L[mask_personal & mask_age_p10] = np.random.uniform(0.0604, 0.0609, size=np.sum(mask_personal & mask_age_p10))

    # Назначение ставок для автокредитов
    mask_age_a3 = loan_ages <= 3  # Возраст до 3 лет
    mask_age_a5 = (loan_ages > 3) & (loan_ages <= 5)  # Возраст 3–5 лет
    mask_age_a10 = (loan_ages > 5) & (loan_ages <= 10)  # Возраст 5–10 лет
    r_L[mask_auto & mask_age_a3] = np.random.uniform(0.0339, 0.0349, size=np.sum(mask_auto & mask_age_a3))
    r_L[mask_auto & mask_age_a5] = np.random.uniform(0.0349, 0.0379, size=np.sum(mask_auto & mask_age_a5))
    r_L[mask_auto & mask_age_a10] = np.random.uniform(0.0379, 0.0399, size=np.sum(mask_auto & mask_age_a10))

    return r_L

# Получение ожидаемых потерь по рейтингам
def get_credit_landa_from_rating_vec(credit_ratings):
    """Векторная функция для получения ожидаемых потерь."""
    lambda_i = np.zeros_like(credit_ratings, dtype=float)  # Массив ожидаемых потерь
    mask_aaa = credit_ratings == 'AAA'  # Маска для рейтинга AAA
    mask_aa = credit_ratings == 'AA'  # Маска для рейтинга AA
    mask_a = credit_ratings == 'A'  # Маска для рейтинга A
    mask_bbb = credit_ratings == 'BBB'  # Маска для рейтинга BBB
    mask_bb = credit_ratings == 'BB'  # Маска для рейтинга BB

    # Назначение λ_i по рейтингам
    lambda_i[mask_aaa] = np.random.uniform(0.0002, 0.0003, size=np.sum(mask_aaa))
    lambda_i[mask_aa] = np.random.uniform(0.0003, 0.001, size=np.sum(mask_aa))
    lambda_i[mask_a] = np.random.uniform(0.001, 0.0024, size=np.sum(mask_a))
    lambda_i[mask_bbb] = np.random.uniform(0.0024, 0.0058, size=np.sum(mask_bbb))
    lambda_i[mask_bb] = np.random.uniform(0.0058, 0.0119, size=np.sum(mask_bb))

    return lambda_i

# Расчет дохода от кредитов
def calc_loan_revenue(df_customers: pd.DataFrame):
    """Рассчет дохода от кредитов."""
    r_L = get_loan_interest_rate_vec(df_customers['Loan Type'].values, df_customers['Loan Age'].values)  # Процентные ставки
    lambda_i = get_credit_landa_from_rating_vec(df_customers['Credit Rating'].values)  # Ожидаемые потери
    valid_mask = ~np.isnan(r_L) & ~np.isnan(lambda_i)  # Маска для валидных данных
    return np.sum(df_customers['Loan Size'].values[valid_mask] * r_L[valid_mask] - lambda_i[valid_mask])  # Доход

# Расчет затрат на кредиты
def calc_loan_cost(df_customers: pd.DataFrame, bank_predetermined_institutional_cost):
    """Рассчет затрат на кредиты."""
    return np.sum(df_customers['Loan Size'].values * bank_predetermined_institutional_cost)  # Затраты

# Расчет транзакционных издержек
def calc_total_transaction_cost(df_customers: pd.DataFrame, bank_required_reserve_ratio, financial_institutions_deposit):
    """Рассчет общей стоимости транзакций."""
    total_loan_size = df_customers['Loan Size'].sum()  # Общий размер кредитов
    K = bank_required_reserve_ratio  # Коэффициент резерва
    D = financial_institutions_deposit  # Депозиты
    T = (1 - K) * D - total_loan_size  # Транзакционные активы
    r_T = 0.01  # Ставка транзакционных издержек
    return r_T * T if T >= 0 else 0  # Стоимость транзакций

# Расчет стоимости депозитов
def calc_cost_of_demand_deposit(rD, financial_institutions_deposit):
    """Рассчет стоимости депозитов."""
    return rD * financial_institutions_deposit  # Стоимость

# Суммирование ожидаемых потерь
def calc_sum_of_landa(df_customers: pd.DataFrame):
    """Суммирование ожидаемых потерь."""
    lambda_i = get_credit_landa_from_rating_vec(df_customers['Credit Rating'].values)  # Ожидаемые потери
    valid_mask = ~np.isnan(lambda_i)  # Маска для валидных данных
    return np.sum(lambda_i[valid_mask])  # Сумма потерь

# Вычисление фитнес-функции
def calc_fitness(df_customers: pd.DataFrame, bank_required_reserve_ratio, financial_institutions_deposit, rD, bank_predetermined_institutional_cost):
    """Фитнес-функция."""
    if df_customers.empty:
        return 0
    v = calc_loan_revenue(df_customers)  # Доход от кредитов
    w_bar = calc_total_transaction_cost(df_customers, bank_required_reserve_ratio, financial_institutions_deposit)  # Транзакционные издержки
    beta = calc_cost_of_demand_deposit(rD, financial_institutions_deposit)  # Стоимость депозитов
    sum_of_landa = calc_sum_of_landa(df_customers)  # Ожидаемые потери
    mue = calc_loan_cost(df_customers, bank_predetermined_institutional_cost)  # Затраты на кредиты
    return v + w_bar - mue - beta - sum_of_landa  # Фитнес

# Проверка ограничений GAMCC
def is_GAMCC_satisfied_vec(chromo, df_customers, bank_required_reserve_ratio, financial_institutions_deposit):
    """Векторная проверка ограничений GAMCC."""
    selected_indices = np.where(chromo)[0]  # Индексы выбранных клиентов
    loan_sizes = df_customers['Loan Size'].values[selected_indices]  # Размеры кредитов
    credit_limits = df_customers['Credit Limit'].values[selected_indices]  # Кредитные лимиты
    total_loan = np.sum(loan_sizes)  # Общий размер кредитов
    return (total_loan <= (1 - bank_required_reserve_ratio) * financial_institutions_deposit and
            np.all(loan_sizes <= credit_limits))  # Соответствие ограничениям

# Категоризация размера кредита
def get_loan_category(loan):
    """Определение категории кредита."""
    if loan <= 13000:
        return 'micro'
    elif 13000 < loan <= 50000:
        return 'small'
    elif 50000 < loan <= 100000:
        return 'medium'
    elif 100000 < loan <= 250000:
        return 'large'
    else:
        return np.nan  # Некорректный размер

# Категоризация по ожидаемым потерям
def get_landa_category(landa):
    """Определение рейтинга по λ_i."""
    if 0.0002 <= landa <= 0.0003:
        return 'AAA'
    elif 0.0003 < landa <= 0.001:
        return 'AA'
    elif 0.001 < landa <= 0.0024:
        return 'A'
    elif 0.0024 < landa <= 0.0058:
        return 'BBB'
    elif 0.0058 < landa <= 0.0119:
        return 'BB'
    else:
        return np.nan  # Некорректное значение

# Категоризация возраста кредита
def get_loan_age_category(loan_age):
    """Определение категории возраста кредита."""
    if 1 <= loan_age <= 3:
        return 1
    elif 3 < loan_age <= 5:
        return 2
    elif 5 < loan_age <= 10:
        return 3
    elif 10 < loan_age <= 20:
        return 4
    else:
        return np.nan  # Некорректный возраст

# Инициализация популяции
def init_population_with_customers(df_customers, customer_size, population_size, bank_required_reserve_ratio, financial_institutions_deposit):
    """Инициализация популяции с учетом GAMCC."""
    chromos = []  # Список хромосом
    max_loan_allowed = (1 - bank_required_reserve_ratio) * financial_institutions_deposit  # Максимальная сумма кредитов
    loan_sizes = df_customers['Loan Size'].values  # Размеры кредитов
    credit_limits = df_customers['Credit Limit'].values  # Кредитные лимиты

    # Создание хромосом
    for _ in range(population_size):
        chromo = np.zeros(customer_size, dtype=bool)  # Пустая хромосома
        num_selected = np.random.randint(1, min(30, customer_size + 1))  # Количество выбираемых клиентов
        indices = np.random.choice(range(customer_size), size=num_selected, replace=False)  # Индексы клиентов
        total_loan = np.sum(loan_sizes[indices])  # Общий размер кредитов
        attempts = 0  # Количество попыток
        max_attempts = 100  # Максимум попыток
        # Проверка ограничений
        while (total_loan > max_loan_allowed or
               np.any(loan_sizes[indices] > credit_limits[indices]) and
               attempts < max_attempts):
            num_selected = np.random.randint(1, min(30, customer_size + 1))
            indices = np.random.choice(range(customer_size), size=num_selected, replace=False)
            total_loan = np.sum(loan_sizes[indices])
            attempts += 1
        if attempts < max_attempts:
            chromo[indices] = True
            chromos.append(chromo)
    return np.array(chromos) if chromos else np.array([])  # Массив хромосом

# Нормировка фитнес-вектора
def get_rated_fit_vector(chromos_fitness_vector):
    """Нормировка фитнес-вектора."""
    fitness = chromos_fitness_vector  # Вектор фитнеса
    if fitness.min() < 0:
        fitness = fitness - fitness.min()  # Сдвиг для устранения отрицательных значений
    sum_fitness = fitness.sum()  # Сумма фитнеса
    return fitness / sum_fitness if sum_fitness > 0 else np.ones(len(fitness)) / len(fitness)  # Нормированные вероятности

# Кроссовер
def xover(parent1: np.ndarray, parent2: np.ndarray):
    """Кроссовер."""
    rand_spliter = np.random.randint(0, len(parent1))  # Точка раздела
    child1 = parent1.copy()  # Первый потомок
    child2 = parent2.copy()  # Второй потомок
    np.put(child1, list(range(rand_spliter, len(parent1))), parent2[rand_spliter:])  # Обмен генами
    np.put(child2, list(range(0, rand_spliter)), parent1[:rand_spliter])
    return child1, child2

# Мутация
def mutate(parent1: np.ndarray, parent2: np.ndarray):
    """Мутация."""
    rand_pos_for_mutate = np.random.randint(0, len(parent1))  # Позиция для мутации
    child1 = parent1.copy()  # Первый потомок
    child2 = parent2.copy()  # Второй потомок
    ch1_gene = True  # Новое значение гена для первого потомка
    ch2_gene = True  # Новое значение гена для второго потомка
    if parent2[rand_pos_for_mutate]:
        ch1_gene = False
    if parent1[rand_pos_for_mutate]:
        ch2_gene = False
    np.put(child1, [rand_pos_for_mutate], [ch1_gene])  # Применение мутации
    np.put(child2, [rand_pos_for_mutate], [ch2_gene])
    return child1, child2

# Селекция методом колеса рулетки
def roulette_wheel_selection(chromos, fitness_vector):
    """Селекция методом взвешенного колеса рулетки."""
    rated_fitness = get_rated_fit_vector(fitness_vector)  # Нормированные вероятности
    selected_indices = np.random.choice(
        range(len(chromos)),
        size=len(chromos),
        replace=True,
        p=rated_fitness
    )  # Выбор индексов
    return selected_indices.tolist()

# Удаление худших хромосом
def delete_chromo_based_on_bad_fit(chromos: np.ndarray, fitness_vector, number_of_chromo_to_be_deleted: int) -> np.ndarray:
    """Удаление худших хромосом."""
    selected_chromos = []  # Индексы хромосом для удаления
    while len(selected_chromos) < number_of_chromo_to_be_deleted:
        min_checked_val = fitness_vector.max()  # Максимальное значение для поиска минимума
        min_found_index = 0  # Индекс минимального фитнеса
        for chromo_index in range(len(fitness_vector)):
            if chromo_index in selected_chromos:
                continue
            if min_checked_val > fitness_vector[chromo_index]:
                min_checked_val = fitness_vector[chromo_index]
                min_found_index = chromo_index
        selected_chromos.append(min_found_index)
    target_chromos_index = [item for item in list(range(0, len(chromos))) if item not in selected_chromos]  # Оставшиеся индексы
    return chromos[target_chromos_index, :]

# Получение данных по хромосоме
def get_dataframe_by_chromo(df_customers: pd.DataFrame, chromo) -> pd.DataFrame:
    """Возвращение данных выбранных клиентов."""
    selected_customers = np.where(chromo)[0]  # Индексы выбранных клиентов
    return df_customers.iloc[selected_customers, :]  # Подмножество данных

# Список колонок для клиентов
def get_cols_customer():
    """Возвращает список колонок для данных клиентов."""
    return ['ID', 'Loan Age', 'Loan Size', 'Loan Type', 'Credit Rating', 'Credit Limit']

# Список колонок для результатов
def get_cols_result():
    """Возвращает список колонок для результатов."""
    return ['M%', 'P%', 'LA%', 'D', 'POP_SIZE', 'AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS', 'GENERATION_SIZE']