'''
Пример использования Q-Learning или StableBaseline3 для обучения пользовательской среды.
'''
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

# Импорт модуля среды, чтобы зарегистрировать её в Gym (не используется непосредственно в коде)
import v0_delivery_robot_env 
import pickle  # Для сохранения и загрузки Q-таблицы
from stable_baselines3 import A2C  # Алгоритм Advantage Actor Critic
import os  # Для работы с файловой системой

# Функция для обучения или тестирования с использованием Q-Learning
def run_q(episodes, is_training=True, render=False):
    # Создание среды 'delivery-robot-v0'
    env = gym.make('delivery-robot-v0', render_mode='human' if render else None)

    if is_training:
        # Инициализация Q-таблицы: 5-мерный массив [позиция робота, позиция цели, действия]
        q = np.zeros((env.unwrapped.grid_rows, env.unwrapped.grid_cols,
                      env.unwrapped.grid_rows, env.unwrapped.grid_cols, env.action_space.n))
    else:
        # Загрузка Q-таблицы из файла при тестировании
        with open('v0_delivery_solution.pkl', 'rb') as f:
            q = pickle.load(f)

    # Гиперпараметры
    learning_rate_a = 0.9   # Альфа, скорость обучения
    discount_factor_g = 0.9 # Гамма, дисконтный фактор. 0 = приоритет текущему состоянию, 1 = приоритет будущему.
    epsilon = 1             # Эпсилон: вероятность выбора случайного действия (1 = 100% случайные действия)

    # Массив для хранения количества шагов, чтобы робот нашел цель
    steps_per_episode = np.zeros(episodes)

    step_count = 0
    for i in range(episodes):
        if render:
            print(f'Episode {i}')  # Печать текущего эпизода

        # Сбрасываем среду в начале эпизода
        state = env.reset()[0]
        terminated = False

        # Робот выполняет действия, пока не найдет цель
        while not terminated:
            # Выбор действия с использованием эпсилон-жадной стратегии
            if is_training and random.random() < epsilon:
                # Выбираем случайное действие
                action = env.action_space.sample()
            else:
                # Преобразуем состояние в индекс для Q-таблицы
                q_state_idx = tuple(state)

                # Выбираем лучшее действие из Q-таблицы
                action = np.argmax(q[q_state_idx])

            # Выполняем действие
            new_state, reward, terminated, _, _ = env.step(action)

            # Преобразуем состояния и действия для индексации в Q-таблице
            q_state_action_idx = tuple(state) + (action,)
            q_new_state_idx = tuple(new_state)

            if is_training:
                # Обновляем Q-таблицу с использованием уравнения Беллмана
                q[q_state_action_idx] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx]
                )

            # Обновляем текущее состояние
            state = new_state

            # Учет шагов
            step_count += 1
            if terminated:
                steps_per_episode[i] = step_count  # Сохраняем количество шагов для текущего эпизода
                step_count = 0

        # Постепенно уменьшаем эпсилон (для уменьшения случайности)
        epsilon = max(epsilon - 1 / episodes, 0)

    env.close()

    # Построение графика среднего количества шагов на 100 эпизодов
    sum_steps = np.zeros(episodes)
    for t in range(episodes):
        sum_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])  # Среднее по последним 100 эпизодам
    plt.plot(sum_steps)
    plt.xlabel('Episodes')
    plt.ylabel('number of steps')
    plt.savefig('v0_delivery_solution.png')

    if is_training:
        # Сохранение Q-таблицы в файл
        with open("v0_delivery_solution.pkl", "wb") as f:
            pickle.dump(q, f)

# Функция для обучения с использованием StableBaseline3
def train_sb3():
    # Папки для хранения моделей и логов
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('delivery-robot-v0')  # Создаем среду

    # Используем алгоритм Advantage Actor Critic (A2C) из StableBaseline3
    model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 500  # Количество шагов на итерацию обучения
    iters = 0
    while True:
        iters += 1

        # Тренируем модель на заданное количество шагов
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        # Сохраняем обученную модель
        model.save(f"{model_dir}/a2c_{TIMESTEPS * iters}")

# Функция для тестирования модели StableBaseline3
def test_sb3(render=True):
    env = gym.make('delivery-robot-v0', render_mode='human' if render else None)

    # Загрузка обученной модели
    model = A2C.load('models/a2c_2000', env=env)

    # Запуск теста
    obs = env.reset()[0]
    terminated = False
    while True:
        # Предсказание действия (детерминированно, чтобы получить одинаковое поведение)
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            break

# Основная точка входа
if __name__ == '__main__':
    # Обучение и тестирование с использованием Q-Learning
    run_q(500, is_training=True, render=False)
    run_q(1, is_training=False, render=True)

    # Обучение и тестирование с использованием StableBaseline3
    # train_sb3()
    # test_sb3()
