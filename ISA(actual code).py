import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import matplotlib
import math

task_data = pd.read_excel('input task file')
matplotlib.rcParams['font.family'] = ['SimSun', 'Times New Roman']
plt.rcParams['mathtext.default'] = 'regular'

J = len(task_data)
r = task_data['r_j'].tolist()
u = task_data['u_j'].tolist()
G_A = task_data['G_A'].tolist()
G_B = task_data['G_B'].tolist()
G_C = task_data['G_C'].tolist()
G_D = task_data['G_D'].tolist()
t_start = task_data['t_start'].tolist()
P = sorted(set(r + u))
K = len(P)
try:
    m = P.index(21.3)
except ValueError:
   m = -1

I = 3
v = 1.5
t_load_constant = 7
S_safe = 2.2
Q = 3
S_total = 270
initial_positions = [-44, -41.8, -39.6, -37.4, -35.2, -33, -30.8, -28.6, -26.4, -24.2, -22, -19.8, -17.6, -15.4, -13.2,
                     -11, -8.8, -6.6, -4.4, -2.2, 0]
t_move_o = []
for i in range(I):
    t_o = (P[0] - initial_positions[i])/v
    t_move_o.append(t_o)

pently_value = J * 100

t_move = [0] * K
for k in range(K):
    if k == 0:
        t_move[k] = (S_total - P[K - 1] + P[k]) / v
    else:
        t_move[k] = (P[k] - P[k-1])/v
print(t_move)

def generate_initial_solution():
    individual = []
    max_attempts = 1000000
    for j in range(J):
        attempts = 0
        valid = False
        while not valid and attempts < max_attempts:
            i = random.randint(1, I)
            q = random.randint(1, Q)
            attempts += 1
            if check_constraints(individual, i, q, j):
                individual.extend([i, q])
                valid = True
        if not valid:
            return []

    return individual

def check_constraints(individual, i, q, j):
    jrgv = []
    gab_count = sum(
        1 for idx in range(j) if individual[2 * idx] == i and individual[2 * idx + 1] == q and (G_A[idx] + G_B[idx]))
    gcd_count = sum(
        1 for idx in range(j) if individual[2 * idx] == i and individual[2 * idx + 1] == q and (G_C[idx] + G_D[idx]))
    gc_count = sum(
        1 for idx in range(j) if individual[2 * idx] == i and individual[2 * idx + 1] == q + 1 and G_C[idx])
    gb_count = sum(
        1 for idx in range(j) if individual[2 * idx] == i and individual[2 * idx + 1] == q - 1 and G_B[idx])
    if gab_count == 1 and (G_A[j] + G_B[j]):
        return False
    if gcd_count == 1 and (G_C[j] + G_D[j]):
        return False
    if gc_count == 1 and G_B[j]:
        return False
    if gb_count == 1 and G_C[j]:
        return False
    for idx in range(j):
        if individual[2 * idx] == i and individual[2 * idx + 1] == q:
            jrgv.append(idx)
            JRGV = len(jrgv)
            if (G_A[idx] + G_B[idx]) == 1 and (G_C[j] + G_D[j]) == 1:
                if r[idx] < u[j]:
                    return False
            elif (G_C[idx] + G_D[idx]) == 1 and (G_A[j] + G_B[j]) == 1:
                if r[j] < u[idx]:
                    return False

    return True

def calculate_gR_jk_gU_jk():
    gR_jk = np.zeros((J, K))
    gU_jk = np.zeros((J, K))
    for j in range(J):
        for k in range(K):
            if r[j] == P[k]:
                gR_jk[j, k] = 1
            if u[j] == P[k]:
                gU_jk[j, k] = 1
    return gR_jk, gU_jk

def calculate_hR_iqkj_hU_iqkj(individual):
    h_iqk = np.zeros((I, Q + 1, K))
    x = np.zeros((I, Q, J))
    gR_jk, gU_jk = calculate_gR_jk_gU_jk()
    for i in range(I):
        for q in range(Q):
            for j in range(J):
                if individual[2 * j] - 1 == i and individual[2 * j + 1] - 1 == q:
                    x[i, q, j] = 1
    for i in range(I):
        for q in range(Q):
            for j in range(J):
                if x[i, q, j] == 1:
                    for k in range(K):
                        if gR_jk[j, k] == 1:
                            h_iqk[i, q, k] = 1
                        if gU_jk[j, k] == 1:
                            h_iqk[i, q + G_D[j], k] = 1
    return h_iqk, gR_jk, gU_jk, x

def calculate_t_load(h_iqk):
    t_load = np.zeros((I, Q + 1, K))
    for q in range(Q + 1):
        for k in range(K):
            if q == Q and k > m:
                return t_load
            for i in range(I):
                if h_iqk[i, q, k] == 1:
                    t_load[i, q, k] = t_load_constant
    return t_load

def calculate_T( individual):
    h_iqk, gR_jk, gU_jk, x = calculate_hR_iqkj_hU_iqkj(individual)
    t_load = calculate_t_load(h_iqk)
    T = np.zeros((I, Q + 1, K))
    t_jam = np.zeros((I, Q + 1, K))
    for q in range(Q + 1):
        for k in range(K):
            if q == Q and k > m:
                return T, t_jam, h_iqk, t_load, gR_jk, gU_jk, x
            for i in range(I - 1, -1, -1):
                if q == 0 and k == 0:
                    if i < I - 1:
                        t_jam[i, q, k] = max((T[i + 1, q, k] - (t_move_o[i] - S_safe / v)), 0)
                    T[i, q, k] = t_move_o[i] + t_jam[i, q, k] + t_load[i, q, k]
                elif q > 0 and k == 0:
                    if i < I - 1:
                        t_jam[i, q, k] = max((T[i + 1, q, k] - (T[i, q - 1, K - 1] + t_move[k] - S_safe / v)), 0)
                    else:
                        t_jam[i, q, k] = max((T[0, q - 1, k] - (T[i, q - 1, K - 1] + t_move[k] - S_safe / v)), 0)
                    T[i, q, k] = T[i, q - 1, K - 1] + t_move[k] + t_jam[i, q, k] + t_load[i, q, k]
                else:
                    if i < I - 1:
                        t_jam[i, q, k] = max((T[i + 1, q, k] - (T[i, q, k - 1] + t_move[k] - S_safe / v)), 0)
                    elif q > 0:
                        t_jam[i, q, k] = max((T[0, q - 1, k] - (T[i, q, k - 1] + t_move[k] - S_safe / v)), 0)
                    T[i, q, k] = T[i, q, k - 1] + t_move[k] + t_jam[i, q, k] + t_load[i, q, k]

    return T, t_jam, h_iqk, t_load, gR_jk, gU_jk, x

def calculate_end_start1_time(T, gR_jk, gU_jk, x):
    t_end = [0] * J
    t_start1 = [0] * J
    j1 = 0
    j2 = 0
    for i in range(I):
        for q in range(Q):
            for j in range(J):
                if x[i, q, j] == 1:
                    for k in range(K):
                        if gR_jk[j, k] == 1:
                            t_start1[j] = T[i, q, k] - t_load_constant
                            j1 += 1
                        if gU_jk[j, k] == 1:
                            t_end[j] = T[i, q + G_D[j], k]
                            j2 += 1
    return t_end, t_start1

def calculate_fitness(individual):
    T, t_jam, h_iqk, t_load, gR_jk, gU_jk, x = calculate_T(individual)
    t_end, t_start1 = calculate_end_start1_time(T, gR_jk, gU_jk, x)
    T_finish = [0] * J
    penalty = 0

    for j in range(J):
        T_finish[j] = t_end[j] - t_start[j]
        if t_start1[j] < t_start[j]:
            penalty += pently_value

    for i in range(1, I + 1):
        for q in range(1, Q + 1):
            ga_count = sum(1 for idx in range(J) if
                            individual[2 * idx] == i and individual[2 * idx + 1] == q and G_A[idx])
            gd_count = sum(1 for idx in range(J) if
                            individual[2 * idx] == i and individual[2 * idx + 1] == q and G_D[idx])
            gb_count = sum(1 for idx in range(J) if
                             individual[2 * idx] == i and individual[2 * idx + 1] == q and G_B[idx])
            gc_count = sum(1 for idx in range(J) if
                             individual[2 * idx] == i and individual[2 * idx + 1] == q and G_C[idx])
            if ga_count + gb_count > 1 or gd_count + gc_count > 1:
                penalty += pently_value

    for i in range(I):
        Liq = [[] for _ in range(Q + 1)]
        for q in range(Q + 1):
            for k in range(K):
                if q == Q and k > m:
                    break
                if t_load[i, q, k] != 0:
                    Liq[q].append(k)
        for q in range(Q):
            LIQ = len(Liq[q])
            for idx in range(LIQ):
                if idx < LIQ - 1:
                    if P[Liq[q][idx]] in r and P[Liq[q][idx+1]] in r:
                        penalty += pently_value
                    if P[Liq[q][idx]] in u and P[Liq[q][idx+1]] in u:
                        penalty += pently_value
                elif Liq[q + 1] != []:
                    if P[Liq[q][idx]] in r and P[Liq[q + 1][0]] in r:
                        penalty += pently_value
                    if P[Liq[q][idx]] in u and P[Liq[q + 1][0]] in u:
                        penalty += pently_value

    return sum(T_finish), penalty

def generate_neighbor1(solution, task):
    neighbor = solution.copy()
    change_type = random.choice(['rgv', 'circle', 'swap'])

    if change_type == 'rgv':
        neighbor[2 * task] = random.randint(1, I)
    elif change_type == 'circle':
        neighbor[2 * task + 1] = random.randint(1, Q)
    else:
        task1 = task
        task2 = random.randint(0, J - 1)
        if task1 != task2:
            neighbor[2 * task1], neighbor[2 * task2] = neighbor[2 * task2], neighbor[2 * task1]
            neighbor[2 * task1 + 1], neighbor[2 * task2 + 1] = neighbor[2 * task2 + 1], neighbor[2 * task1 + 1]

    return neighbor, change_type

def build_precomp():
    group_AB = [j for j in range(J) if (G_A[j] + G_B[j] == 1)]
    group_CD = [j for j in range(J) if (G_C[j] + G_D[j] == 1)]

    sorted_AB_by_t = sorted(group_AB, key=lambda x: t_start[x])
    sorted_CD_by_t = sorted(group_CD, key=lambda x: t_start[x])

    pre = {
        "group_by_t": {
            "AB": sorted_AB_by_t,
            "CD": sorted_CD_by_t
        },
        "which_group": {},
        "q_block_of_j": {},
        "block_tasks_of_j": {},
        "i_choice_of_j": {}
    }

    def fill_group_info(group_name, group_sorted_by_t):
        n = len(group_sorted_by_t)
        for block_start in range(0, n, I):
            block_end = min(block_start + I, n)
            block_tasks = group_sorted_by_t[block_start:block_end]
            q_block = min((block_start // I) + 1, Q)
            block_sorted_by_r = sorted(block_tasks, key=lambda x: (-r[x], t_start[x]))
            i_order = list(range(I, 0, -1))[:len(block_tasks)]
            for k, j in enumerate(block_sorted_by_r):
                pre["which_group"][j] = group_name
                pre["q_block_of_j"][j] = q_block
                pre["block_tasks_of_j"][j] = block_tasks
                pre["i_choice_of_j"][j] = i_order[k]

    fill_group_info("AB", sorted_AB_by_t)
    fill_group_info("CD", sorted_CD_by_t)
    return pre
def generate_neighbor2(solution, task, pre):
    neighbor = solution.copy()
    change_type = random.choice(['rgv', 'circle', 'swap'])

    if change_type == 'rgv':
        new_i = pre["i_choice_of_j"][task]
        neighbor[2 * task] = new_i

    elif change_type == 'circle':
        q_block = pre["q_block_of_j"][task]
        neighbor[2 * task + 1] = q_block

    else:
        new_i = pre["i_choice_of_j"][task]
        q_block = pre["q_block_of_j"][task]
        neighbor[2 * task] = new_i
        neighbor[2 * task + 1] = q_block

    return neighbor, change_type

def simulated_annealing(INITIAL_TEMPERATURE, FINAL_TEMPERATURE, COOLING_RATE1, COOLING_RATE2, ITERATIONS_PER_TEMP, T_yuzhi):
    max_init_attempts = 10
    for attempt in range(max_init_attempts):
        current_solution = generate_initial_solution()
        if current_solution:
            break
    else:
        print("无法生成有效的初始解")
        return [], 0, 0, []

    start_time = time.time()
    t_sum, penalty = calculate_fitness(current_solution)
    current_fitness = t_sum + penalty

    best_solution = current_solution.copy()
    best_fitness = current_fitness

    fitness_history = [best_fitness]
    temperature = INITIAL_TEMPERATURE
    iteration = 0
    c = 1
    a = 0
    temperature_total = []
    print(f"初始适应度: {current_fitness}")
    b2 = 0
    COOLING_RATE = COOLING_RATE1
    ITERATIONS_PER_TEMP = ITERATIONS_PER_TEMP
    pre = build_precomp()
    print(pre)
    elements = [{"id": j, "q": pre["q_block_of_j"][j], "r": r[j]} for j in range(J)]

    sorted_elements = sorted(elements, key=lambda x: (x["q"], x["r"]))
    print(sorted_elements)

    sorted_ids = [element["id"] for element in sorted_elements]
    print(sorted_ids)
    while temperature > FINAL_TEMPERATURE:
        b1 = 0
        for _ in range(int(ITERATIONS_PER_TEMP)):
            task = sorted_ids[_ % J]
            if random.random() < 0.8-0.8*(a/(math.log(((-1 / math.log(COOLING_RATE1)) + T_yuzhi)/INITIAL_TEMPERATURE, COOLING_RATE1)+
                                           math.log((-1 / math.log(COOLING_RATE1)) / ((-1 / math.log(COOLING_RATE1)) + T_yuzhi),COOLING_RATE2)+
                                           math.log(FINAL_TEMPERATURE / (-1 / math.log(COOLING_RATE1)), COOLING_RATE1))):
                neighbor_solution, change_type = generate_neighbor2(current_solution, task, pre)
            else:
                neighbor_solution, change_type = generate_neighbor1(current_solution, task)
            try:
                t_sum_neighbor,  penalty_neighbor = calculate_fitness(neighbor_solution)
                neighbor_fitness = t_sum_neighbor + penalty_neighbor
            except:
                continue
            delta = neighbor_fitness - current_fitness
            if delta < 0:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                b2 += 1
                b1 += 1

                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness

            else:
                probability = math.exp(- delta / temperature)
                if random.random() < probability:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    b2 += 1
                    b1 += 1

            iteration += 1
            fitness_history.append(best_fitness)

            if iteration % ITERATIONS_PER_TEMP == 0:
                print(f"迭代 {iteration}: 当前适应度 = {current_fitness:.2f}, "
                      f"最优适应度 = {best_fitness:.2f}, 温度 = {temperature:.4f},降温率 = {COOLING_RATE:.2f}")
            temperature_total.append(temperature)

        if temperature < (-1/math.log(COOLING_RATE1)) + T_yuzhi and temperature > (-1/math.log(COOLING_RATE1)):
            COOLING_RATE = COOLING_RATE2
        else:
            COOLING_RATE = COOLING_RATE1

        temperature *= COOLING_RATE
        a += 1

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"模拟退火算法完成")
    print(f"总执行时间: {execution_time:.4f} 秒")
    print(f"最优适应度: {best_fitness}")

    return best_solution, best_fitness, execution_time, fitness_history

INITIAL_TEMPERATURE = 1000
FINAL_TEMPERATURE = 0.1
COOLING_RATE1 = 0.94
COOLING_RATE2 = 0.96
ITERATIONS_PER_TEMP = 2 * J
T_yuzhi = 500

simulated_annealing(INITIAL_TEMPERATURE, FINAL_TEMPERATURE, COOLING_RATE1, COOLING_RATE2, ITERATIONS_PER_TEMP, T_yuzhi)

