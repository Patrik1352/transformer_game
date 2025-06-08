import pygame
import sys
import math
import time
from typing import List, Tuple, Dict

# Инициализация Pygame
pygame.init()

# Константы
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MENU_WIDTH = 300
BORDER_WIDTH = 3
BORDER_RADIUS = 10

# Эталонные последовательности для энкодера и декодера
ENCODER_SEQUENCE = [
    "Input\nEmbedding",
    "Positional\nEncoding",
    "Multi-Head\nAttention",
    "Add & Norm",
    "Feed\nForward",
    "Add & Norm"
]

DECODER_SEQUENCE = [
    "Output\nEmbedding",
    "Positional\nEncoding",
    "Masked\nMulti-Head\nAttention",
    "Add & Norm",
    "Multi-Head\nAttention",
    "Add & Norm",
    "Feed\nForward",
    "Add & Norm",
    "Linear",
    "Softmax"
]

# Соединения для энкодера
ENCODER_CONNECTIONS = [
    (0, 1),  # Input Embedding -> Positional Encoding
    (1, 2),  # Positional Encoding -> Multi-Head Attention
    (2, 3),  # Multi-Head Attention -> Add & Norm
    (1, 3),  # Positional Encoding -> Add & Norm
    (3, 5),  # Multi-Head Attention -> Add & Norm
    (3, 4),  # Add & Norm -> Feed Forward
    (4, 5)   # Feed Forward -> Add & Norm
]

# Соединения для декодера
DECODER_CONNECTIONS = [
    (0, 1),  # Output Embedding -> Positional Encoding
    (1, 2),  # Positional Encoding -> Masked Multi-Head Attention
    (1, 3),  # Positional Encoding -> Add & Norm
    (2, 3),  # Masked Multi-Head Attention -> Add & Norm
    (3, 4),  # Add & Norm -> Multi-Head Attention
    (3, 5),  # Add & Norm -> Add & Norm
    (4, 5),  # Multi-Head Attention -> Add & Norm
    (5, 6),  # Add & Norm -> Feed Forward
    (6, 7),  # Feed Forward -> Add & Norm
    (5, 7),  # Add & Norm -> Add & Norm
    (7, 8),  # Add & Norm -> Linear
    (8, 9),   # Linear -> Softmax
]

DECODER_SEQUENCE = [
    "Output\nEmbedding",
    "Positional\nEncoding",
    "Masked\nMulti-Head\nAttention",
    "Add & Norm",
    "Multi-Head\nAttention",
    "Add & Norm",
    "Feed\nForward",
    "Add & Norm",
    "Linear",
    "Softmax"
]

COLORS = {
    'Input\nEmbedding': (247, 225, 225),
    'Output\nEmbedding': (247, 225, 225),
    'Attention': (250, 227, 192),
    '''Multi-Head\nAttention''': (250, 227, 192),
    'Masked\nMulti-Head\nAttention': (250, 227, 192),
    'Add & Norm': (242, 244, 198),
    'Feed\nForward': (201, 231, 245),
    'Softmax': (209, 230, 209),
    'Linear': (220, 223, 238),
    'text': (0,0,0),
    'WHITE': (255, 255, 255),
    'Positional\nEncoding': (255, 255, 255),
    'MENU_BG': (240, 240, 240),
    'TRASH_BG': (255, 200, 200),
    'BORDER': (0, 0, 0),
    'ERROR': (255, 0, 0),  # Красный цвет для ошибок
    'CHECK_BUTTON': (100, 200, 100)  # Зеленый цвет для кнопки проверки
}

SIZE = {
    'pos': (50,50),
    'block': (150, 50),
    'small_block': (150, 25),  # высота в 2 раза меньше стандартного
    'big_block': (150, 75)  # высота в 1.5 раза больше стандартного
}

class ConnectionPoint:
    def __init__(self, block, side: str):
        self.block = block
        self.side = side  # 'top', 'right', 'bottom', 'left'
        self.radius = 5
        self.update_position()
        self.visible = False
        self.hover_time = 0
        self.hover_threshold = 0.5  # время в секундах для появления точки

    def update_position(self):
        if self.side == 'top':
            self.pos = (self.block.rect.centerx, self.block.rect.top)
        elif self.side == 'right':
            self.pos = (self.block.rect.right, self.block.rect.centery)
        elif self.side == 'bottom':
            self.pos = (self.block.rect.centerx, self.block.rect.bottom)
        else:  # left
            self.pos = (self.block.rect.left, self.block.rect.centery)

    def draw(self, screen):
        if self.visible:
            pygame.draw.circle(screen, COLORS['BORDER'], self.pos, self.radius)

    def is_clicked(self, pos):
        if not self.visible:
            return False
        return math.sqrt((pos[0] - self.pos[0])**2 + (pos[1] - self.pos[1])**2) <= self.radius

    def show(self):
        self.visible = True
        self.hover_time = 0

    def hide(self):
        self.visible = False
        self.hover_time = 0

class Arrow:
    def __init__(self, start_point: ConnectionPoint, end_point: ConnectionPoint):
        self.start_point = start_point
        self.end_point = end_point
        self.width = 3

    def draw(self, screen):
        pygame.draw.line(screen, COLORS['BORDER'], 
                        self.start_point.pos, self.end_point.pos, 
                        self.width)
        # Рисуем стрелку
        angle = math.atan2(self.end_point.pos[1] - self.start_point.pos[1],
                          self.end_point.pos[0] - self.start_point.pos[0])
        arrow_length = 10
        arrow_angle = math.pi / 6  # 30 градусов
        
        # Вычисляем точки стрелки
        arrow_point1 = (
            self.end_point.pos[0] - arrow_length * math.cos(angle - arrow_angle),
            self.end_point.pos[1] - arrow_length * math.sin(angle - arrow_angle)
        )
        arrow_point2 = (
            self.end_point.pos[0] - arrow_length * math.cos(angle + arrow_angle),
            self.end_point.pos[1] - arrow_length * math.sin(angle + arrow_angle)
        )
        
        # Рисуем стрелку
        pygame.draw.polygon(screen, COLORS['BORDER'], 
                          [self.end_point.pos, arrow_point1, arrow_point2])

    def is_point_near(self, point, threshold=10):
        # Вычисляем расстояние от точки до линии
        x1, y1 = self.start_point.pos
        x2, y2 = self.end_point.pos
        x, y = point
        
        # Вычисляем длину линии
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return False
            
        # Вычисляем проекцию точки на линию
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length**2)))
        
        # Вычисляем координаты проекции
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        # Вычисляем расстояние от точки до проекции
        distance = math.sqrt((x - projection_x)**2 + (y - projection_y)**2)
        
        return distance <= threshold

class TransformerBlock:
    def __init__(self, name: str, pos: Tuple[int, int],
                 size: Tuple[int, int] = (SIZE['block'][0], SIZE['block'][0])):
        self.name = name
        self.pos = pos
        self.size = size
        self.dragging = False
        self.connections = []
        self.color = COLORS.get(name, (100,100,100))
        self.rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        # Добавляем точки соединения
        self.connection_points = {
            'top': ConnectionPoint(self, 'top'),
            'right': ConnectionPoint(self, 'right'),
            'bottom': ConnectionPoint(self, 'bottom'),
            'left': ConnectionPoint(self, 'left')
        }
        self.is_hovered = False
        self.hover_start_time = 0

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, screen):
        # Рисуем закругленный прямоугольник с обводкой
        pygame.draw.rect(screen, self.color, self.rect, border_radius=BORDER_RADIUS)
        pygame.draw.rect(screen, COLORS['BORDER'], self.rect, width=BORDER_WIDTH, border_radius=BORDER_RADIUS)
        
        # Отрисовка текста
        font = pygame.font.Font(None, 24)
        lines = self.name.split('\n')
        line_height = font.get_height()
        total_height = line_height * len(lines)
        
        for i, line in enumerate(lines):
            text = font.render(line, True, COLORS['text'])
            text_rect = text.get_rect(centerx=self.rect.centerx,
                                    centery=self.rect.centery - total_height/2 + line_height/2 + i*line_height)
            screen.blit(text, text_rect)

        # Обновляем и отрисовываем точки соединения
        for point in self.connection_points.values():
            point.update_position()
            point.draw(screen)

    def move(self, pos):
        self.rect.x = pos[0] - self.size[0] // 2
        self.rect.y = pos[1] - self.size[1] // 2

    def check_hover(self, pos, is_connecting=False):
        if is_connecting:
            for point in self.connection_points.values():
                point.show()
            return

        current_time = time.time()
        is_currently_hovered = self.rect.collidepoint(pos)

        if is_currently_hovered and not self.is_hovered:
            # Только что навели на блок
            self.is_hovered = True
            self.hover_start_time = current_time
        elif not is_currently_hovered and self.is_hovered:
            # Вышли за пределы блока
            self.is_hovered = False
            for point in self.connection_points.values():
                point.hide()
        elif is_currently_hovered and self.is_hovered:
            # Находимся внутри блока
            if current_time - self.hover_start_time >= 0.5:
                for point in self.connection_points.values():
                    point.show()

class YinYangBlock:
    def __init__(self, name: str, pos: Tuple[int, int],
                 size: Tuple[int, int] = (SIZE['block'][0], SIZE['block'][0])):
        self.name = name
        self.pos = pos
        self.size = size
        self.dragging = False
        self.connections = []
        self.color = COLORS.get('WHITE', (100,100,100))
        self.rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
        self.radius = min(size[0], size[1]) // 2 - 5  # радиус с небольшим отступом
        self.center = (pos[0] + size[0]//2, pos[1] + size[1]//2)
        # Добавляем точки соединения
        self.connection_points = {
            'top': ConnectionPoint(self, 'top'),
            'right': ConnectionPoint(self, 'right'),
            'bottom': ConnectionPoint(self, 'bottom'),
            'left': ConnectionPoint(self, 'left')
        }
        self.is_hovered = False
        self.hover_start_time = 0

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def check_hover(self, pos, is_connecting=False):
        if is_connecting:
            for point in self.connection_points.values():
                point.show()
            return

        current_time = time.time()
        is_currently_hovered = self.rect.collidepoint(pos)

        if is_currently_hovered and not self.is_hovered:
            # Только что навели на блок
            self.is_hovered = True
            self.hover_start_time = current_time
        elif not is_currently_hovered and self.is_hovered:
            # Вышли за пределы блока
            self.is_hovered = False
            for point in self.connection_points.values():
                point.hide()
        elif is_currently_hovered and self.is_hovered:
            # Находимся внутри блока
            if current_time - self.hover_start_time >= 0.5:
                for point in self.connection_points.values():
                    point.show()

    def draw(self, screen):
        # Рисуем Инь-Янь
        x, y = self.center
        h = 2  # толщина линии обводки

        # Рисуем основную окружность
        pygame.draw.circle(screen, COLORS['BORDER'], self.center, self.radius, h)

        # Рисуем правую полуокружность (верхняя часть после поворота)
        pygame.draw.arc(screen, COLORS['BORDER'], 
                       (x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius), 
                       0, math.pi, h)

        # Рисуем левую полуокружность (нижняя часть после поворота)
        pygame.draw.arc(screen, COLORS['BORDER'], 
                       (x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius), 
                       math.pi, 2 * math.pi, h)

        # Рисуем правый малый полукруг (верхний после поворота)
        small_r = self.radius // 2
        pygame.draw.arc(screen, COLORS['BORDER'], 
                       (x - small_r * 2, y - small_r, small_r * 2, small_r * 2), 
                       0, math.pi, h)

        # Рисуем левый малый полукруг (нижний после поворота)
        pygame.draw.arc(screen, COLORS['BORDER'],
                       (x, y - small_r, small_r * 2, small_r * 2), 
                       math.pi, 2 * math.pi, h)

        # Обновляем и отрисовываем точки соединения
        for point in self.connection_points.values():
            point.update_position()
            point.draw(screen)

        font = pygame.font.Font(None, 24)
        lines = self.name.split('\n')
        line_height = font.get_height()
        total_height = line_height * len(lines)

        for i, line in enumerate(lines):
            text = font.render(line, True, COLORS['text'])
            text_rect = text.get_rect(centerx=self.rect.centerx-75,
                                      centery=self.rect.centery - total_height / 2 + line_height / 2 + i * line_height)
            screen.blit(text, text_rect)

        # Обновляем и отрисовываем точки соединения
        for point in self.connection_points.values():
            point.update_position()
            point.draw(screen)

    def move(self, pos):
        self.rect.x = pos[0] - self.size[0] // 2
        self.rect.y = pos[1] - self.size[1] // 2
        self.center = (self.rect.x + self.size[0]//2, self.rect.y + self.size[1]//2)

class Menu:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, MENU_WIDTH, WINDOW_HEIGHT)
        self.blocks = []
        block_menu = [
            ("Input\nEmbedding", SIZE['block']),
            ('Output\nEmbedding', SIZE['block']),
            ("Multi-Head\nAttention", SIZE['block']),
            ("Masked\nMulti-Head\nAttention", SIZE['big_block']),
            ("Add & Norm", SIZE['small_block']),
            ("Linear", SIZE['small_block']),
            ("Feed\nForward", SIZE['block']),
            ("Softmax", SIZE['small_block'])
        ]
        spacing = 10
        current_y = 20
        for name, size in block_menu:
            self.blocks.append(TransformerBlock(name,
                                                ((MENU_WIDTH-size[0])//2, current_y),
                                                size))
            current_y += size[1] + spacing

        self.blocks.append(YinYangBlock("Positional\nEncoding", ((MENU_WIDTH-SIZE['pos'][0])//2+50, current_y), SIZE['pos']))

        self.trash_rect = pygame.Rect(100, WINDOW_HEIGHT - 100, 150, 50)
        self.trash_img = pygame.image.load("static/trash.png")
        self.trash_img = pygame.transform.smoothscale(self.trash_img, (self.trash_rect.width//1.5, self.trash_rect.height))
        
        # Добавляем кнопку проверки
        self.check_button_rect = pygame.Rect(50, WINDOW_HEIGHT - 200, 200, 50)
        self.check_button_font = pygame.font.Font(None, 36)
        self.check_button_text = "Проверить"
        self.check_button_text_surface = self.check_button_font.render(self.check_button_text, True, COLORS['text'])
        self.check_button_text_rect = self.check_button_text_surface.get_rect(center=self.check_button_rect.center)

    def draw(self, screen):
        # Фон меню
        pygame.draw.rect(screen, COLORS['MENU_BG'], self.rect)
        
        # Отрисовка блоков в меню
        for block in self.blocks:
            block.draw(screen)
        
        # Отрисовка корзины-иконки
        screen.blit(self.trash_img, self.trash_rect)
        
        # Отрисовка кнопки проверки
        pygame.draw.rect(screen, COLORS['CHECK_BUTTON'], self.check_button_rect, border_radius=BORDER_RADIUS)
        screen.blit(self.check_button_text_surface, self.check_button_text_rect)

    def is_in_trash(self, pos):
        return self.trash_rect.collidepoint(pos)
        
    def is_check_button_clicked(self, pos):
        return self.check_button_rect.collidepoint(pos)

    def get_block_at_pos(self, pos):
        for block in self.blocks:
            if block.is_clicked(pos):
                return block
        return None

class TransformerGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Архитектура Трансформера")
        self.clock = pygame.time.Clock()
        self.menu = Menu()
        self.blocks = []
        self.encoder_blocks = []  # Сохраняем блоки энкодера
        self.encoder_arrows = []  # Сохраняем стрелки энкодера
        self.selected_block = None
        self.dragging = False
        self.dragging_from_menu = False
        self.arrows = []
        self.connecting = False
        self.start_connection_point = None
        self.error_blocks = set()
        self.error_arrows = set()
        self.message_font = pygame.font.Font(None, 36)
        self.message = None
        self.message_timer = 0
        self.current_mode = 'encoder'  # 'encoder' или 'decoder'
        self.encoder_decoder_connected = False  # Флаг соединения энкодера и декодера

    def check_sequence(self):
        # Сбрасываем предыдущие ошибки
        self.error_blocks.clear()
        self.error_arrows.clear()
        
        # Определяем текущую последовательность и соединения
        if self.current_mode == 'encoder':
            reference_sequence = ENCODER_SEQUENCE
            reference_connections = ENCODER_CONNECTIONS
        else:
            reference_sequence = DECODER_SEQUENCE
            reference_connections = DECODER_CONNECTIONS
        
        # Проверяем количество элементов
        if len(self.blocks) != len(reference_sequence):
            self.show_message(f"Неверное количество элементов для {self.current_mode}!")
            return False
            
        # Сортируем блоки по их позиции (снизу вверх, слева направо)
        sorted_blocks = sorted(self.blocks, key=lambda b: (-b.rect.y, b.rect.x))
        
        # Проверяем каждый блок
        for i, block in enumerate(sorted_blocks):
            if block.name != reference_sequence[i]:
                self.error_blocks.add(block)
                self.show_message(f"Неверный элемент на позиции {i+1} в {self.current_mode}: {block.name}")
                return False
                
        # Проверяем соединения
        for start_idx, end_idx in reference_connections:
            start_block = sorted_blocks[start_idx]
            end_block = sorted_blocks[end_idx]
            
            # Ищем соответствующую стрелку
            connection_found = False
            for arrow in self.arrows:
                if (arrow.start_point.block == start_block and 
                    arrow.end_point.block == end_block):
                    connection_found = True
                    break
                    
            if not connection_found:
                self.show_message(f"Отсутствует соединение между {start_block.name.replace('\n',' ')} и {end_block.name.replace('\n',' ')} в {self.current_mode}")
                return False
        
        # Если все проверки пройдены успешно
        if self.current_mode == 'encoder':
            self.current_mode = 'decoder'
            self.encoder_blocks = self.blocks.copy()  # Сохраняем блоки энкодера
            self.encoder_arrows = self.arrows.copy()  # Сохраняем стрелки энкодера
            self.show_message("Энкодер собран правильно! Теперь соберите декодер")
            self.blocks.clear()  # Очищаем только текущие блоки
            self.arrows.clear()  # Очищаем только текущие стрелки
        else:
            # Проверяем соединение между энкодером и декодером
            encoder_last_block = self.encoder_blocks[-1]  # Последний блок энкодера
            decoder_mha_block = None
            for block in self.blocks:
                if block.name == "Multi-Head\nAttention":
                    decoder_mha_block = block
                    break
            
            if not decoder_mha_block:
                self.show_message("Не найден блок Multi-Head Attention в декодере!")
                return False
                
            # Проверяем наличие соединения между энкодером и декодером
            encoder_decoder_connected = False
            for arrow in self.arrows:
                if (arrow.start_point.block == encoder_last_block and 
                    arrow.end_point.block == decoder_mha_block):
                    encoder_decoder_connected = True
                    break
            
            if not encoder_decoder_connected:
                self.show_message("Отлично! Осталось соедините энкодер и декодер!")
                return False
                
            self.show_message("Все верно! Вы шикарны!")
        
        return True

    def show_message(self, text):
        self.message = text
        self.message_timer = 180  # Показывать сообщение 3 секунды (60 FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    # Проверяем клик по кнопке проверки
                    if self.menu.is_check_button_clicked(event.pos):
                        self.check_sequence()
                        return True
                        
                    # Проверяем клик по корзине
                    if self.menu.is_in_trash(event.pos):
                        if self.current_mode == 'encoder':
                            self.blocks.clear()
                        else:
                            self.blocks.clear()
                            self.encoder_blocks.clear()
                            self.encoder_arrows.clear()
                        self.arrows.clear()
                        self.error_blocks.clear()
                        self.error_arrows.clear()
                        return True

                    # Проверяем клик по меню
                    if event.pos[0] < MENU_WIDTH:
                        menu_block = self.menu.get_block_at_pos(event.pos)
                        if menu_block:
                            # Создаем новый блок того же типа, что и в меню
                            if isinstance(menu_block, YinYangBlock):
                                new_block = YinYangBlock(
                                    menu_block.name,
                                    (event.pos[0], event.pos[1]),
                                    menu_block.size
                                )
                            else:
                                new_block = TransformerBlock(
                                    menu_block.name,
                                    (event.pos[0], event.pos[1]),
                                    menu_block.size
                                )
                            self.blocks.append(new_block)
                            self.selected_block = new_block
                            self.dragging = True
                            self.dragging_from_menu = True
                    else:
                        # Проверяем клик по точкам соединения
                        if self.current_mode == 'decoder':
                            # Проверяем точки соединения как в блоках декодера, так и в блоках энкодера
                            for block in self.blocks + self.encoder_blocks:
                                for point in block.connection_points.values():
                                    if point.is_clicked(event.pos):
                                        self.connecting = True
                                        self.start_connection_point = point
                                        break
                                if self.connecting:
                                    break
                        else:
                            # В режиме энкодера проверяем только точки в текущих блоках
                            for block in self.blocks:
                                for point in block.connection_points.values():
                                    if point.is_clicked(event.pos):
                                        self.connecting = True
                                        self.start_connection_point = point
                                        break
                                if self.connecting:
                                    break
                        
                        # Если не кликнули по точке соединения, проверяем клик по блокам
                        if not self.connecting:
                            for block in self.blocks:
                                if block.is_clicked(event.pos):
                                    self.selected_block = block
                                    self.dragging = True
                                    self.dragging_from_menu = False
                                    break

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if self.dragging and self.selected_block:
                        if self.menu.is_in_trash(event.pos):
                            self.blocks.remove(self.selected_block)
                    self.dragging = False
                    self.selected_block = None
                    self.dragging_from_menu = False
                    
                    # Завершаем соединение
                    if self.connecting:
                        if self.current_mode == 'decoder':
                            # В режиме декодера проверяем точки как в блоках декодера, так и в блоках энкодера
                            for block in self.blocks + self.encoder_blocks:
                                for point in block.connection_points.values():
                                    if point.is_clicked(event.pos) and point != self.start_connection_point:
                                        self.arrows.append(Arrow(self.start_connection_point, point))
                                        break
                        else:
                            # В режиме энкодера проверяем только точки в текущих блоках
                            for block in self.blocks:
                                for point in block.connection_points.values():
                                    if point.is_clicked(event.pos) and point != self.start_connection_point:
                                        self.arrows.append(Arrow(self.start_connection_point, point))
                                        break
                        self.connecting = False
                        self.start_connection_point = None

            if event.type == pygame.MOUSEMOTION:
                if self.dragging and self.selected_block:
                    if not self.dragging_from_menu or event.pos[0] >= MENU_WIDTH:
                        self.selected_block.move(event.pos)
                
                # Проверяем наведение на блоки
                for block in self.blocks:
                    block.check_hover(event.pos, self.connecting)
                if self.current_mode == 'decoder':
                    for block in self.encoder_blocks:
                        block.check_hover(event.pos, self.connecting)

        return True

    def draw(self):
        self.screen.fill(COLORS['WHITE'])
        
        # Отрисовка фоновых прямоугольников
        gray_color = (243, 243, 244)  # Серый цвет
        rect1 = pygame.Rect(400, 300, 250, 350)
        rect2 = pygame.Rect(700, 100, 250, 550)
        pygame.draw.rect(self.screen, gray_color, rect1, border_radius=BORDER_RADIUS)
        pygame.draw.rect(self.screen, gray_color, rect2, border_radius=BORDER_RADIUS)
        
        # Отрисовка меню
        self.menu.draw(self.screen)
        
        # Отрисовка стрелок энкодера, если мы в режиме декодера
        if self.current_mode == 'decoder':
            for arrow in self.encoder_arrows:
                arrow.draw(self.screen)
        
        # Отрисовка стрелок
        for arrow in self.arrows:
            color = COLORS['ERROR'] if arrow in self.error_arrows else COLORS['BORDER']
            arrow.draw(self.screen)
        
        # Отрисовка блоков энкодера, если мы в режиме декодера
        if self.current_mode == 'decoder':
            for block in self.encoder_blocks:
                block.draw(self.screen)
        
        # Отрисовка всех блоков
        for block in self.blocks:
            block.draw(self.screen)
            # Если блок в списке ошибок, рисуем его название красным
            if block in self.error_blocks:
                font = pygame.font.Font(None, 24)
                lines = block.name.split('\n')
                line_height = font.get_height()
                total_height = line_height * len(lines)
                
                for i, line in enumerate(lines):
                    text = font.render(line, True, COLORS['ERROR'])
                    text_rect = text.get_rect(centerx=block.rect.centerx,
                                            centery=block.rect.centery - total_height/2 + line_height/2 + i*line_height)
                    self.screen.blit(text, text_rect)

        # Отрисовка временной линии соединения
        if self.connecting and self.start_connection_point:
            pygame.draw.line(self.screen, COLORS['BORDER'],
                           self.start_connection_point.pos,
                           pygame.mouse.get_pos(),
                           2)

        # Отрисовка сообщения
        if self.message and self.message_timer > 0:
            text_surface = self.message_font.render(self.message, True, COLORS['text'])
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH//2, 50))
            self.screen.blit(text_surface, text_rect)
            self.message_timer -= 1

        # Отрисовка текущего режима
        mode_text = f"Собери блок {self.current_mode}"
        mode_surface = self.message_font.render(mode_text, True, COLORS['text'])
        mode_rect = mode_surface.get_rect(center=(WINDOW_WIDTH//2, 30))
        self.screen.blit(mode_surface, mode_rect)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = TransformerGame()
    game.run() 