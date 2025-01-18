import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env  

import v0_delivery_robot as wr  # Импорт класса DeliveryRobot
import numpy as np

# Регистрация этого модуля как среды Gym
register(
    id='delivery-robot-v0',  # Уникальный идентификатор среды
    entry_point='v0_delivery_robot_env:DeliveryRobotEnv'  # Точка входа в класс DeliveryRobotEnv
)

# Класс, описывающий среду для симуляции работы робота доставки
class DeliveryRobotEnv(gym.Env):
    # Метаданные, включая режим отображения и FPS
    metadata = {'render_modes': ['human'], 'render_fps': 7}

    def __init__(self, grid_rows=4, grid_cols=5, obstacles=None, render_mode=None):
        # Размер сетки
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode  # Режим отображения (например, 'human')

        # Опционально задаем препятствия
        self.obstacles = obstacles if obstacles else []

        # Инициализация объекта DeliveryRobot
        self.delivery_robot = wr.DeliveryRobot(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=self.metadata['render_fps'],
            obstacles=self.obstacles
        )

        # Gym требует определить пространство действий (дискретное: количество действий)
        self.action_space = spaces.Discrete(len(wr.RobotAction))

        # Пространство наблюдений: состояния робота и цели
        self.observation_space = spaces.Box(
            low=0,  # Минимальные значения (левая верхняя ячейка сетки)
            high=np.array([self.grid_rows - 1, self.grid_cols - 1, self.grid_rows - 1, self.grid_cols - 1]),  # Максимальные значения (правая нижняя ячейка сетки)
            shape=(4,),  # Форма: 4 значения (позиция робота и цели)
            dtype=np.int32  # Тип данных: целые числа
        )

    # Метод для сброса среды
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Сброс случайного генератора
        self.delivery_robot.reset(seed=seed)  # Сброс робота доставки

        # Обновляем список препятствий
        self.current_obstacles = self.delivery_robot.obstacles.copy()

        # Конструируем наблюдение: объединяем позиции робота и цели
        obs = np.concatenate((self.delivery_robot.robot_pos, self.delivery_robot.target_pos))

        info = {}  # Дополнительная информация

        # Отображаем среду, если режим 'human'
        if self.render_mode == 'human':
            self.render()

        return obs, info  # Возвращаем наблюдение и информацию

    # Метод для выполнения действия в среде
    def step(self, action):
        # Выполнение действия роботом и проверка достижения цели
        target_reached = self.delivery_robot.perform_action(wr.RobotAction(action))

        # Проверка столкновения с препятствием
        robot_pos = self.delivery_robot.robot_pos
        if tuple(robot_pos) in self.current_obstacles:
            reward = -1  # Штраф за столкновение
            terminated = True  # Завершаем эпизод
        else:
            reward = 0  # Нет награды или штрафа
            terminated = False
            if target_reached:
                reward = 1  # Награда за достижение цели
                terminated = True  # Завершаем эпизод

        # Формируем наблюдение: объединяем позиции робота и цели
        obs = np.concatenate((self.delivery_robot.robot_pos, self.delivery_robot.target_pos))

        info = {}  # Дополнительная информация

        # Отображаем действие и визуализируем среду, если режим 'human'
        if self.render_mode == 'human':
            print(wr.RobotAction(action))  # Печатаем действие
            self.render()

        return obs, reward, terminated, False, info  # Возвращаем результат

    # Метод для визуализации
    def render(self):
        # Отображаем сетку с препятствиями, роботом и целью
        self.delivery_robot.render()
        # print("Obstacles at positions:", self.current_obstacles)  # Отображение позиций препятствий

# Код для тестирования класса
if __name__ == "__main__":
    # Создаем объект среды
    env = gym.make('delivery-robot-v0', render_mode='human')

    # Сбрасываем среду и получаем начальное наблюдение
    obs = env.reset()[0]

    # Цикл действий
    while True:
        # Случайное действие из пространства действий
        rand_action = env.action_space.sample()
        # Выполняем действие и получаем результат
        obs, reward, terminated, _, _ = env.step(rand_action)

        # Если эпизод завершен, сбрасываем среду
        if terminated:
            obs = env.reset()[0]