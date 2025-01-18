import random
from enum import Enum
import pygame
import sys
from os import path

# Перечисление возможных действий робота
class RobotAction(Enum):
    LEFT = 0   # Движение влево
    DOWN = 1   # Движение вниз
    RIGHT = 2  # Движение вправо
    UP = 3     # Движение вверх

# Перечисление возможных типов плиток на сетке
class GridTile(Enum):
    _FLOOR = 0     # Пол (свободная клетка)
    ROBOT = 1      # Позиция робота
    TARGET = 2     # Цель (пакет)
    OBSTACLE = 3   # Препятствие

    # Переопределение метода __str__, чтобы выводить первую букву имени
    def __str__(self):
        return self.name[:1]

# Класс для управления роботом в складе
class DeliveryRobot:
    def __init__(self, grid_rows=4, grid_cols=5, fps=5, obstacles=None):
        # Размеры сетки (строки и столбцы)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # Список препятствий (по умолчанию пуст)
        self.obstacles = obstacles if obstacles else []
        # Сброс параметров игры
        self.reset()
        # Частота кадров в секунду (FPS)
        self.fps = fps
        # Последнее действие робота
        self.last_action = ''
        # Инициализация Pygame
        self._init_pygame()

    # Метод для инициализации графики с помощью Pygame
    def _init_pygame(self):
        pygame.init()
        pygame.display.init()

        self.clock = pygame.time.Clock()
        # Шрифт для отображения действий
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        # Размеры ячейки в пикселях
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)

        # Размер окна
        self.window_size = (
            self.cell_width * self.grid_cols,
            self.cell_height * self.grid_rows + self.action_info_height
        )

        # Создание окна для отображения
        self.window_surface = pygame.display.set_mode(self.window_size)

        # Загрузка изображений для графического отображения элементов сетки
        file_name = path.join(path.dirname(__file__), "sprites/bot_blue.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/package.png")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/obstacle.png")
        img = pygame.image.load(file_name)
        self.obstacle_img = pygame.transform.scale(img, self.cell_size)

    # Метод для сброса игры (инициализация позиций)
    def reset(self, seed=None):
        self.robot_pos = [0, 0]  # Позиция робота начинается с (0, 0)

        random.seed(seed)

        # Генерация случайной позиции цели
        while True:
            self.target_pos = [
                random.randint(1, self.grid_rows - 1),
                random.randint(1, self.grid_cols - 1)
            ]
            if self.target_pos != self.robot_pos:  # Убедимся, что цель не совпадает с позицией робота
                break

        # Генерация случайных препятствий, которые не перекрывают робота и цель
        self.obstacles = []
        for _ in range(4):  # Количество препятствий (можно изменить)
            while True:
                obstacle = [
                    random.randint(0, self.grid_rows - 1),
                    random.randint(0, self.grid_cols - 1)
                ]
                if (
                    obstacle != self.robot_pos and
                    obstacle != self.target_pos and
                    obstacle not in self.obstacles
                ):
                    self.obstacles.append(obstacle)
                    break

        # Проверка, существует ли путь от робота до цели
        while not self._is_path_clear():
            self.reset(seed)  # Если путь недоступен, повторяем генерацию

    # Метод для проверки доступности пути от робота до цели
    def _is_path_clear(self):
        from collections import deque

        visited = set()
        queue = deque([tuple(self.robot_pos)])

        # Направления движения: вверх, вниз, влево, вправо
        directions = [
            (-1, 0), (1, 0),  # Вверх, вниз
            (0, -1), (0, 1)   # Влево, вправо
        ]

        while queue:
            current = queue.popleft()
            if current == tuple(self.target_pos):
                return True  # Если достигли цели, путь найден

            for dr, dc in directions:
                new_pos = (current[0] + dr, current[1] + dc)

                if (
                    0 <= new_pos[0] < self.grid_rows and
                    0 <= new_pos[1] < self.grid_cols and
                    new_pos not in visited and
                    list(new_pos) not in self.obstacles
                ):
                    visited.add(new_pos)
                    queue.append(new_pos)

        return False  # Путь не найден

    # Метод для выполнения действия робота
    def perform_action(self, robot_action: RobotAction) -> bool:
        self.last_action = robot_action
        new_pos = self.robot_pos[:]

        # Определение нового положения робота в зависимости от действия
        if robot_action == RobotAction.LEFT:
            new_pos[1] -= 1
        elif robot_action == RobotAction.RIGHT:
            new_pos[1] += 1
        elif robot_action == RobotAction.UP:
            new_pos[0] -= 1
        elif robot_action == RobotAction.DOWN:
            new_pos[0] += 1

        # Проверка, является ли новое положение допустимым
        if self.is_valid_position(new_pos):
            self.robot_pos = new_pos

        # Возвращает True, если робот достиг цели
        return self.robot_pos == self.target_pos

    # Метод для проверки допустимости положения
    def is_valid_position(self, position):
        if position[0] < 0 or position[0] >= self.grid_rows or position[1] < 0 or position[1] >= self.grid_cols:
            return False  # Положение вне границ сетки

        if position in self.obstacles:  # Положение совпадает с препятствием
            return False
        return True

    # Метод для визуализации сетки
    def render(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if [r, c] == self.robot_pos:
                    print(GridTile.ROBOT, end=' ')
                elif [r, c] == self.target_pos:
                    print(GridTile.TARGET, end=' ')
                elif [r, c] in self.obstacles:
                    print(GridTile.OBSTACLE, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')
            print()
        print()

        self._process_events()

        # Очистка экрана
        self.window_surface.fill((255, 255, 255))

        # Отображение всех элементов сетки
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if [r, c] == self.target_pos:
                    self.window_surface.blit(self.goal_img, pos)

                if [r, c] == self.robot_pos:
                    self.window_surface.blit(self.robot_img, pos)

                if [r, c] in self.obstacles:
                    self.window_surface.blit(self.obstacle_img, pos)

        # Отображение информации о последнем действии
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0, 0, 0), (255, 255, 255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)

        # Обновление дисплея
        pygame.display.update()
        self.clock.tick(self.fps)

    # Обработка событий Pygame (например, закрытие окна)
    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

# Основная программа
if __name__ == "__main__":
    # Инициализация объекта DeliveryRobot
    deliveryRobot = DeliveryRobot()
    deliveryRobot.render()

    # Запуск бесконечного цикла для выполнения случайных действий роботом
    while True:
        # Случайное действие выбирается из списка RobotAction
        rand_action = random.choice(list(RobotAction))
        print(rand_action)

        # Выполнение действия и обновление визуализации
        deliveryRobot.perform_action(rand_action)
        deliveryRobot.render()
