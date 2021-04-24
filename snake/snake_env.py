# -*- coding: utf-8 -*-

import pygame
import random
import os

import numpy as np

WHITE = (0xff, 0xff, 0xff)
BLACK = (0, 0, 0)
GREEN = (0, 0xff, 0)
RED = (0xff, 0, 0)
BLUE = (0, 0, 0xff)
LINE_COLOR = (0x00, 0x00, 0x00)
FPS = 30

HARD_LEVEL = list(range(4, FPS // 2, 2))
hardness = HARD_LEVEL[0]

D_LEFT, D_RIGHT, D_UP, D_DOWN = 0, 1, 2, 3

WIDTH, HEIGHT = 500, 500
SCREEN_WIDTH = WIDTH + 600

# 贪吃蛇小方块的宽度
CUBE_WIDTH = 20

# 计算屏幕的网格数，网格的大小就是小蛇每一节身体的大小
GRID_WIDTH_NUM, GRID_HEIGHT_NUM = int(WIDTH / CUBE_WIDTH),\
                                int(HEIGHT / CUBE_WIDTH)


class GameStatus(object):
    pass


class SnakeEnv(object):
    left = D_LEFT
    right = D_RIGHT
    up = D_UP
    down = D_DOWN

    def __init__(self,
                 need_render=True,
                 use_simple=False,
                 alg='NN遗传算法',
                 set_life=None,
                 no_sight_disp=False):
        self.need_render = need_render
        self.use_simple = use_simple
        self.n_moves = 0
        self.alg = alg
        self.set_life = set_life
        self.no_sight_disp = no_sight_disp
        self.reset_game_status()
        if need_render:
            self.pygame_init()

    def pygame_init(self):
        # 初始化
        pygame.init()

        # 要想载入音乐，必须要初始化 mixer
        pygame.mixer.init()

        # 设置画布
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, HEIGHT))

        # 设置标题
        pygame.display.set_caption("贪吃蛇")

        # 设置游戏的根目录为当前文件夹
        self.base_folder = os.path.dirname(__file__)

        # 这里需要在当前目录下创建一个名为music的目录，并且在里面存放名为back.mp3的背景音乐
        self.music_folder = os.path.join(self.base_folder, 'music')

        # 背景音乐
        self.back_music = pygame.mixer.music.load(
            os.path.join(self.music_folder, 'back.mp3'))

        # 小蛇吃食物的音乐
        self.bite_sound = pygame.mixer.Sound(
            os.path.join(self.music_folder, 'armor-light.wav'))

        # 图片
        img_folder = os.path.join(self.base_folder, 'images')
        back_img = pygame.image.load(os.path.join(img_folder, 'back.png'))
        self.snake_head_img = pygame.image.load(
            os.path.join(img_folder, 'head.png'))
        self.snake_head_img.set_colorkey(BLACK)
        food_img = pygame.image.load(os.path.join(img_folder, 'orb2.png'))

        # 调整图片的大小，和屏幕一样大
        self.background = pygame.transform.scale(back_img, (WIDTH, HEIGHT))

        self.food = pygame.transform.scale(food_img, (CUBE_WIDTH, CUBE_WIDTH))

        # 设置一下音量大小，防止过大
        pygame.mixer.music.set_volume(0.1)

        # 设置音乐循环次数 -1 表示无限循环
        pygame.mixer.music.play(loops=-1)

        pygame.draw.rect(self.screen, BLACK,
                         (WIDTH, 0, SCREEN_WIDTH - WIDTH, HEIGHT))
        self.draw_grids()
        self.render()

    # 重置所有的状态为初始态
    def reset_game_status(self):
        self.status = GameStatus()

        # 每次小蛇身体加长的时候，我们就将身体的位置加到列表末尾
        h = (GRID_WIDTH_NUM // 2, GRID_HEIGHT_NUM // 2)
        self.status.snake_body = [
            h,
            (h[0] + 1, h[1]),
            # (h[0] + 2, h[1]),
            # (h[0] + 3, h[1]),
        ]

        self.status.food_pos = self.generate_food(self.status)
        self.status.direction = D_LEFT
        self.status.game_is_over = False
        self.status.running = True
        self.status.hardness = HARD_LEVEL[0]
        self.status.score = 0
        if self.set_life:
            self.life = self.set_life
        else:
            self.life = GRID_WIDTH_NUM

        return self.status

    def draw_connection(self, start, end, color=GREEN, width=3):
        start_row, start_col = start // GRID_WIDTH_NUM, start % GRID_WIDTH_NUM
        end_row, end_col = end // GRID_WIDTH_NUM, end % GRID_WIDTH_NUM
        pygame.draw.line(self.screen,
                         color, (start_col * CUBE_WIDTH + CUBE_WIDTH / 2,
                                 start_row * CUBE_WIDTH + CUBE_WIDTH / 2),
                         (end_col * CUBE_WIDTH + CUBE_WIDTH / 2,
                          end_row * CUBE_WIDTH + CUBE_WIDTH / 2),
                         width=width)

    # 画出网格线
    def draw_grids(self):
        for i in range(GRID_WIDTH_NUM):
            pygame.draw.line(self.screen, LINE_COLOR, (i * CUBE_WIDTH, 0),
                             (i * CUBE_WIDTH, HEIGHT))

        for i in range(GRID_HEIGHT_NUM):
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * CUBE_WIDTH),
                             (WIDTH, i * CUBE_WIDTH))

        pygame.draw.line(self.screen, WHITE, (WIDTH, 0), (WIDTH, HEIGHT))

    def draw_sight(self):
        if self.no_sight_disp:
            return

        sight = self.get_simple_status()
        directions = [
            (-1, -1),  # left up
            (0, -1),  # up
            (1, -1),  # right up
            (1, 0),  # right
            (1, 1),  # right down
            (0, 1),  # down
            (-1, 1),  # left down
            (-1, 0),  # left
        ]

        color = [GREEN, BLUE, RED]
        head = self.status.snake_body[0]
        start_point = head[0] * CUBE_WIDTH + CUBE_WIDTH // 2, head[
            1] * CUBE_WIDTH + CUBE_WIDTH // 2

        for i in range(8):
            length = 0
            if sight[i * 3] != 0:
                c = color[0]
                length = 1 / sight[i * 3]
            elif sight[i * 3 + 1] != 0:
                c = color[1]
                length = 1 / sight[i * 3 + 1]
            else:
                c = color[2]
                length = 1 / sight[i * 3 + 2]

            length -= 1
            dst = (
                head[0] + int(directions[i][0] * length),
                head[1] + int(directions[i][1] * length),
            )

            dst = (dst[0] * CUBE_WIDTH + CUBE_WIDTH // 2,
                   dst[1] * CUBE_WIDTH + CUBE_WIDTH // 2)

            pygame.draw.line(self.screen, c, start_point, dst, width=2)

    # 打印身体的函数
    def draw_body(self, status):
        for sb in status.snake_body[1:]:
            self.screen.blit(self.food,
                             (sb[0] * CUBE_WIDTH, sb[1] * CUBE_WIDTH))

        if status.direction == D_LEFT:
            rot = 0
        elif status.direction == D_RIGHT:
            rot = 180
        elif status.direction == D_UP:
            rot = 270
        elif status.direction == D_DOWN:
            rot = 90
        new_head_img = pygame.transform.rotate(self.snake_head_img, rot)
        head = pygame.transform.scale(new_head_img, (CUBE_WIDTH, CUBE_WIDTH))
        self.screen.blit(head, (status.snake_body[0][0] * CUBE_WIDTH,
                                status.snake_body[0][1] * CUBE_WIDTH))

    # 随机产生一个事物
    def generate_food(self, status=None):
        if hasattr(status, 'snake_body') and len(status.snake_body) > (
                GRID_HEIGHT_NUM * GRID_WIDTH_NUM) // 4 * 3:
            total_set = set(range(GRID_HEIGHT_NUM * GRID_WIDTH_NUM))
            exists = set(
                [p[0] + p[1] * GRID_WIDTH_NUM for p in status.snake_body])
            remaining = total_set - exists
            if not remaining:
                return None

            c = np.random.choice(list(remaining))
            return c % GRID_WIDTH_NUM, c // GRID_WIDTH_NUM

        while True:
            pos = (random.randint(0, GRID_WIDTH_NUM - 1),
                   random.randint(0, GRID_HEIGHT_NUM - 1))

            if status is None:
                return pos

            # 如果当前位置没有小蛇的身体，我们就跳出循环，返回食物的位置
            if pos not in status.snake_body:
                return pos

    # 画出食物的主体
    def draw_food(self, statis):
        if not self.status.food_pos:
            return
        self.screen.blit(
            self.food,
            (self.status.food_pos[0] * CUBE_WIDTH,
             self.status.food_pos[1] * CUBE_WIDTH, CUBE_WIDTH, CUBE_WIDTH))

    # 判断贪吃蛇是否吃到了事物，如果吃到了我们就加长小蛇的身体
    def grow(self, status):
        if status.snake_body[0] == status.food_pos:
            # 每次吃到食物，就播放音效
            if self.need_render:
                self.bite_sound.play()
            return True

        return False

    def show_text(self, surf, text, size, x, y, color=WHITE):
        font_name = os.path.join(self.base_folder, 'font/font.ttc')
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()

        text_rect.midleft = (x, y)
        surf.blit(text_surface, text_rect)

    def show_welcome(self, screen):
        self.show_text(screen, u'欢乐贪吃蛇', 30, WIDTH / 2, HEIGHT / 2)
        self.show_text(screen, u'按任意键开始游戏', 20, WIDTH / 2, HEIGHT / 2 + 50)

    def show_dead(self):
        surf = self.screen
        text = u'挂'
        size = 150
        x = WIDTH / 2
        y = HEIGHT / 2
        font_name = os.path.join(self.base_folder, 'font/font.ttc')
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, (0xff, 0xaa, 0xaa))
        text_rect = text_surface.get_rect()

        text_rect.center = (x, y)
        surf.blit(text_surface, text_rect)

    def show_scores(
            self,
            screen,
            status,
            extras,
            start_height=4,
    ):
        height = start_height
        self.show_text(
            screen,
            u'算法: {}'.format(self.alg),
            CUBE_WIDTH,
            WIDTH + CUBE_WIDTH * 3,
            CUBE_WIDTH * height,
        )
        height += 2

        self.show_text(
            screen,
            u'级别: {}'.format(status.hardness),
            CUBE_WIDTH,
            WIDTH + CUBE_WIDTH * 3,
            CUBE_WIDTH * height,
        )
        height += 2

        self.show_text(
            screen,
            u'得分: {}'.format(status.score),
            CUBE_WIDTH,
            WIDTH + CUBE_WIDTH * 3,
            CUBE_WIDTH * height,
        )
        height += 2

        self.show_text(
            screen,
            u'剩余生命: {:.1f}'.format(self.life),
            CUBE_WIDTH,
            WIDTH + CUBE_WIDTH * 3,
            CUBE_WIDTH * height,
        )
        height += 2

        for info in extras:
            self.show_text(screen,
                           info,
                           int(CUBE_WIDTH * 1.5),
                           WIDTH + CUBE_WIDTH * 3,
                           CUBE_WIDTH * (height + 10),
                           color=(0xBB, 0xBB, 0xBB))
        height += 2

        return height

    def reset(self):
        self.status = self.reset_game_status()
        return self.get_status(
        ) if not self.use_simple else self.get_simple_status()

    def show(self):
        maps = self.get_status()
        for i in range(GRID_WIDTH_NUM):
            for j in range(GRID_HEIGHT_NUM):
                x = maps[i, j]
                if x == 0:
                    print('.', end='')
                elif x == 1:
                    print('*', end='')
                elif x == -1:
                    print('X', end='')
                elif x == 2:
                    print('O', end='')

            print('')

    def get_status(self):
        maps = np.zeros((GRID_HEIGHT_NUM, GRID_WIDTH_NUM))
        head = self.status.snake_body[0]
        if 0 <= head[0] < GRID_HEIGHT_NUM and 0 <= head[1] < GRID_WIDTH_NUM:
            maps[head[1], head[0]] = 2
        for sb in self.status.snake_body[1:]:
            maps[sb[1], sb[0]] = 1

        if self.status.food_pos:
            maps[self.status.food_pos[1], self.status.food_pos[0]] = -1

        return maps

    def get_simple_status(self):
        directions = [
            (-1, -1),  # left up
            (0, -1),  # up
            (1, -1),  # right up
            (1, 0),  # right
            (1, 1),  # right down
            (0, 1),  # down
            (-1, 1),  # left down
            (-1, 0),  # left
        ]
        status = []
        for d in directions:
            status.extend(self.look(d))

        return np.array(status)

    def look(self, direction):
        head = self.status.snake_body[0]
        # food distance, body distance, wall distance
        res = [0, 0, 0]
        dist = 0
        body_not_found = True
        pos = (head[0], head[1])
        while True:
            dist += 1
            value = 1 / dist
            pos = (
                pos[0] + direction[0],
                pos[1] + direction[1],
            )
            if not (0 <= pos[0] < GRID_WIDTH_NUM
                    and 0 <= pos[1] < GRID_HEIGHT_NUM):
                res[-1] = value
                return res

            if pos == self.status.food_pos:
                res[0] = value

            if body_not_found and pos in self.status.snake_body:
                body_not_found = False
                res[1] = value

    def run(self, action):
        if self.status.game_is_over:
            return 0, (self.get_status() if not self.use_simple else
                       self.get_simple_status()), True, ''

        self.status.direction = action
        status = self.status
        last_pos = status.snake_body[-1]
        for i in range(len(status.snake_body) - 1, 0, -1):
            status.snake_body[i] = status.snake_body[i - 1]

        # 改变头部的位置
        if action == D_UP:
            status.snake_body[0] = (status.snake_body[0][0],
                                    status.snake_body[0][1] - 1)
        elif action == D_DOWN:
            status.snake_body[0] = (status.snake_body[0][0],
                                    status.snake_body[0][1] + 1)
        elif action == D_LEFT:
            status.snake_body[0] = (status.snake_body[0][0] - 1,
                                    status.snake_body[0][1])
        elif action == D_RIGHT:
            status.snake_body[0] = (status.snake_body[0][0] + 1,
                                    status.snake_body[0][1])

        # 限制小蛇的活动范围
        if status.snake_body[0][0] < 0 or \
            status.snake_body[0][0] >= GRID_WIDTH_NUM or\
            status.snake_body[0][1] < 0 or\
                status.snake_body[0][1] >= GRID_HEIGHT_NUM:

            # 超出屏幕之外游戏结束
            status.game_is_over = True
            if self.need_render:
                self.show_dead()
                pygame.display.update()

        # 限制小蛇不能碰到自己的身体
        for sb in status.snake_body[1:]:
            # 身体的其他部位如果和蛇头（snake_body[0]）重合就死亡
            if sb == status.snake_body[0]:
                status.game_is_over = True
                if self.need_render:
                    self.show_dead()
                    pygame.display.update()

        # 判断小蛇是否吃到了事物，吃到了就成长
        got_food = self.grow(status)

        # 如果吃到了食物我们就产生一个新的食物
        reward = 0.
        if got_food:
            reward = status.hardness
            status.score += reward
            status.food_pos = self.generate_food(status)
            status.snake_body.append(last_pos)
            status.hardness = HARD_LEVEL[min(
                len(status.snake_body) // 10,
                len(HARD_LEVEL) - 1)]
            self.life += GRID_WIDTH_NUM / 10 * 3

        self.life -= 0.1
        if self.life <= 0:
            self.status.game_is_over = True
            reward = -10.

        if self.status.game_is_over:
            self.life = 0.
            reward = -10.

        self.n_moves += 1
        return reward, (self.get_status() if not self.use_simple else
                        self.get_simple_status()), self.status.game_is_over, ''

    def __call__(self, action):
        return self.run(action=action)

    def render(self, *args, blit=True, update=True):
        if blit:
            self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(self.screen, BLACK,
                         (WIDTH, 0, SCREEN_WIDTH - WIDTH, HEIGHT))
        self.draw_grids()
        # 画小蛇的身体
        self.draw_body(self.status)
        # 画出食物
        self.draw_food(self.status)
        self.show_scores(self.screen, self.status, extras=args)
        self.draw_sight()
        if self.status.game_is_over:
            self.show_dead()

        if update:
            pygame.display.update()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = SnakeEnv()
    try:
        env.run()
    except Exception as e:
        print(e)
    finally:
        env.close()
