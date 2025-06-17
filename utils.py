import pandas as pd
import numpy as np
import random

def get_loan_interest_rate_vec(loan_types, loan_ages):
    """Векторная функция для получения процентных ставок."""
    r_L = np.zeros_like(loan_types, dtype=float)
    mask_mortgage = loan_types == 'Mortgage'
    mask_personal = loan_types == 'Personal'
    mask_auto = loan_types == 'Auto'

    # Mortgage
    mask_age_mortgage = (10 < loan_ages) & (loan_ages <= 20)
    r_L[mask_mortgage & mask_age_mortgage] = np.random.uniform(0.021, 0.028, size=np.sum(mask_mortgage & mask_age_mortgage))

    # Personal
    mask_age_p3 = loan_ages <= 3
    mask_age_p5 = (loan_ages > 3) & (loan_ages <= 5)
    mask_age_p10 = (loan_ages > 5) & (loan_ages <= 10)
    r_L[mask_personal & mask_age_p3] = np.random.uniform(0.0599, 0.0601, size=np.sum(mask_personal & mask_age_p3))
    r_L[mask_personal & mask_age_p5] = np.random.uniform(0.0601, 0.0604, size=np.sum(mask_personal & mask_age_p5))
    r_L[mask_personal & mask_age_p10] = np.random.uniform(0.0604, 0.0609, size=np.sum(mask_personal & mask_age_p10))

    # Auto
    mask_age_a3 = loan_ages <= 3
    mask_age_a5 = (loan_ages > 3) & (loan_ages <= 5)
    mask_age_a10 = (loan_ages > 5) & (loan_ages <= 10)
    r_L[mask_auto & mask_age_a3] = np.random.uniform(0.0339, 0.0349, size=np.sum(mask_auto & mask_age_a3))
    r_L[mask_auto & mask_age_a5] = np.random.uniform(0.0349, 0.0379, size=np.sum(mask_auto & mask_age_a5))
    r_L[mask_auto & mask_age_a10] = np.random.uniform(0.0379, 0.0399, size=np.sum(mask_auto & mask_age_a10))

    return r_L

def get_credit_landa_from_rating_vec(credit_ratings):
    """Векторная функция для получения ожидаемых потерь."""
    lambda_i = np.zeros_like(credit_ratings, dtype=float)
    mask_aaa = credit_ratings == 'AAA'
    mask_aa = credit_ratings == 'AA'
    mask_a = credit_ratings == 'A'
    mask_bbb = credit_ratings == 'BBB'
    mask_bb = credit_ratings == 'BB'

    lambda_i[mask_aaa] = np.random.uniform(0.0002, 0.0003, size=np.sum(mask_aaa))
    lambda_i[mask_aa] = np.random.uniform(0.0003, 0.001, size=np.sum(mask_aa))
    lambda_i[mask_a] = np.random.uniform(0.001, 0.0024, size=np.sum(mask_a))
    lambda_i[mask_bbb] = np.random.uniform(0.0024, 0.0058, size=np.sum(mask_bbb))
    lambda_i[mask_bb] = np.random.uniform(0.0058, 0.0119, size=np.sum(mask_bb))

    return lambda_i

def calc_loan_revenue(df_customers: pd.DataFrame):
    """Рассчет дохода от кредитов."""
    r_L = get_loan_interest_rate_vec(df_customers['Loan Type'].values, df_customers['Loan Age'].values)
    lambda_i = get_credit_landa_from_rating_vec(df_customers['Credit Rating'].values)
    valid_mask = ~np.isnan(r_L) & ~np.isnan(lambda_i)
    return np.sum(df_customers['Loan Size'].values[valid_mask] * r_L[valid_mask] - lambda_i[valid_mask])

def calc_loan_cost(df_customers: pd.DataFrame, bank_predetermined_institutional_cost):
    """Рассчет затрат на кредиты."""
    return np.sum(df_customers['Loan Size'].values * bank_predetermined_institutional_cost)

def calc_total_transaction_cost(df_customers: pd.DataFrame, bank_required_reserve_ratio, financial_institutions_deposit):
    """Рассчет общей стоимости транзакций."""
    total_loan_size = df_customers['Loan Size'].sum()
    K = bank_required_reserve_ratio
    D = financial_institutions_deposit
    T = (1 - K) * D - total_loan_size
    r_T = 0.01
    return r_T * T if T >= 0 else 0

def calc_cost_of_demand_deposit(rD, financial_institutions_deposit):
    """Рассчет стоимости депозитов."""
    return rD * financial_institutions_deposit

def calc_sum_of_landa(df_customers: pd.DataFrame):
    """Суммирование ожидаемых потерь."""
    lambda_i = get_credit_landa_from_rating_vec(df_customers['Credit Rating'].values)
    valid_mask = ~np.isnan(lambda_i)
    return np.sum(lambda_i[valid_mask])

def calc_fitness(df_customers: pd.DataFrame, bank_required_reserve_ratio, financial_institutions_deposit, rD, bank_predetermined_institutional_cost):
    """Фитнес-функция."""
    if df_customers.empty:
        return 0
    v = calc_loan_revenue(df_customers)
    w_bar = calc_total_transaction_cost(df_customers, bank_required_reserve_ratio, financial_institutions_deposit)
    beta = calc_cost_of_demand_deposit(rD, financial_institutions_deposit)
    sum_of_landa = calc_sum_of_landa(df_customers)
    mue = calc_loan_cost(df_customers, bank_predetermined_institutional_cost)
    return v + w_bar - mue - beta - sum_of_landa

def is_GAMCC_satisfied_vec(chromo, df_customers, bank_required_reserve_ratio, financial_institutions_deposit):
    """Векторная проверка ограничений GAMCC."""
    selected_indices = np.where(chromo)[0]
    loan_sizes = df_customers['Loan Size'].values[selected_indices]
    credit_limits = df_customers['Credit Limit'].values[selected_indices]
    total_loan = np.sum(loan_sizes)
    return (total_loan <= (1 - bank_required_reserve_ratio) * financial_institutions_deposit and
            np.all(loan_sizes <= credit_limits))

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
        return np.nan

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
        return np.nan

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
        return np.nan

def init_population_with_customers(df_customers, customer_size, population_size, bank_required_reserve_ratio, financial_institutions_deposit):
    """Инициализация популяции с учетом GAMCC."""
    chromos = []
    max_loan_allowed = (1 - bank_required_reserve_ratio) * financial_institutions_deposit
    loan_sizes = df_customers['Loan Size'].values
    credit_limits = df_customers['Credit Limit'].values

    for _ in range(population_size):
        chromo = np.zeros(customer_size, dtype=bool)
        num_selected = np.random.randint(1, min(30, customer_size + 1))
        indices = np.random.choice(range(customer_size), size=num_selected, replace=False)
        total_loan = np.sum(loan_sizes[indices])
        attempts = 0
        max_attempts = 100
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
    return np.array(chromos) if chromos else np.array([])

def get_rated_fit_vector(chromos_fitness_vector):
    """Нормировка фитнес-вектора."""
    fitness = chromos_fitness_vector
    if fitness.min() < 0:
        fitness = fitness - fitness.min()
    sum_fitness = fitness.sum()
    return fitness / sum_fitness if sum_fitness > 0 else np.ones(len(fitness)) / len(fitness)

def xover(parent1: np.ndarray, parent2: np.ndarray):
    """Кроссовер."""
    rand_spliter = np.random.randint(0, len(parent1))
    child1 = parent1.copy()
    child2 = parent2.copy()
    np.put(child1, list(range(rand_spliter, len(parent1))), parent2[rand_spliter:])
    np.put(child2, list(range(0, rand_spliter)), parent1[:rand_spliter])
    return child1, child2

def mutate(parent1: np.ndarray, parent2: np.ndarray):
    """Мутация."""
    rand_pos_for_mutate = np.random.randint(0, len(parent1))
    child1 = parent1.copy()
    child2 = parent2.copy()
    ch1_gene = True
    ch2_gene = True
    if parent2[rand_pos_for_mutate]:
        ch1_gene = False
    if parent1[rand_pos_for_mutate]:
        ch2_gene = False
    np.put(child1, [rand_pos_for_mutate], [ch1_gene])
    np.put(child2, [rand_pos_for_mutate], [ch2_gene])
    return child1, child2

def tournament_selection(chromos, fitness_vector, tournament_size=3):
    """Турнирный отбор."""
    selected_indices = []
    for _ in range(len(chromos)):
        tournament = random.sample(range(len(chromos)), tournament_size)
        winner = max(tournament, key=lambda idx: fitness_vector[idx])
        selected_indices.append(winner)
    return selected_indices

def delete_chromo_based_on_bad_fit(chromos: np.ndarray, fitness_vector, number_of_chromo_to_be_deleted: int) -> np.ndarray:
    """Удаление худших хромосом."""
    selected_chromos = []
    while len(selected_chromos) < number_of_chromo_to_be_deleted:
        min_checked_val = fitness_vector.max()
        min_found_index = 0
        for chromo_index in range(len(fitness_vector)):
            if chromo_index in selected_chromos:
                continue
            if min_checked_val > fitness_vector[chromo_index]:
                min_checked_val = fitness_vector[chromo_index]
                min_found_index = chromo_index
        selected_chromos.append(min_found_index)
    target_chromos_index = [item for item in list(range(0, len(chromos))) if item not in selected_chromos]
    return chromos[target_chromos_index, :]

def get_dataframe_by_chromo(df_customers: pd.DataFrame, chromo) -> pd.DataFrame:
    """Возвращение данных выбранных клиентов."""
    selected_customers = np.where(chromo)[0]
    return df_customers.iloc[selected_customers, :]