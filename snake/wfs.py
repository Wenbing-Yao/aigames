# -*- coding: utf-8 -*-

from time import sleep
import numpy as np
from snake_env import SnakeEnv

from collections import deque
import os
# import pickle


def BFS(mat, src, dest):

    # check source and destination cell
    # of the matrix have value 1
    if mat[src.x][src.y] != 1 or mat[dest.x][dest.y] != 1:
        return -1

    visited = np.zeros_like(mat)

    # Mark the source cell as visited
    visited[src.x][src.y] = True

    # Create a queue for BFS
    q = deque()

    # Distance of source cell is 0
    s = (
        src,
        0,
    )
    q.append(s)
    rowNum = [-1, 0, 0, 1]
    colNum = [0, -1, 1, 0]

    while q:

        curr = q.popleft()  # Dequeue the front cell

        pt = curr[0]
        if pt[0] == dest[0] and pt[1] == dest[1]:
            return curr[1]

        # Otherwise enqueue its adjacent cells
        for i in range(4):
            row = pt[0] + rowNum[i]
            col = pt[1] + colNum[i]

            # if adjacent cell is valid, has path
            # and not visited yet, enqueue it.
            if 0 <= row < mat.shape[0] and 0 <= col < mat.shape[1]\
                    and mat[row][col] == 1 and not visited[row][col]:

                visited[row][col] = True
                Adjcell = (
                    (
                        row,
                        col,
                    ),
                    curr.dist + 1,
                )
                q.append(Adjcell)

    return -1


def bfs(grid, start, dst):
    queue = deque([[start]])
    seen = set([start])
    height, width = grid.shape
    wall = [1, 2]
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (y, x) == dst:
            return path
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][
                    x2] not in wall and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def play(record=0, no_render=False):
    env = SnakeEnv(need_render=not no_render, alg='最短路径')
    obs = env.reset()
    env.render()
    input()
    x, y = [], []

    directions = {
        (-1, 0): env.right,
        (1, 0): env.left,
        (0, -1): env.down,
        (0, 1): env.up
    }

    need_record = True if record else False
    new_dst = None
    origin_dst = None
    # counter = 20
    use_random = False
    while True:
        if not record and not no_render:
            env.render()
        src = np.where(obs == 2)
        src = int(src[1]), int(src[0])
        dst = np.where(obs == -1)
        dst = int(dst[0]), int(dst[1])

        if new_dst is not None:
            paths = bfs(obs, start=src, dst=new_dst)
        else:
            paths = bfs(obs, start=src, dst=dst)

        if paths is None:
            # origin_dst = dst
            # new_dst = (
            #     np.random.randint(0, obs.shape[0]),
            #     np.random.randint(0, obs.shape[1]),
            # )
            # counter -= 1
            # if counter <= 0:
            #     print('score: ', env.status.score)
            #     new_dst = None
            #     origin_dst = None
            #     counter = 20
            #     obs = env.reset()
            # continue
            use_random = True
        else:
            new_dst = None
            if new_dst is not None and paths[1] == new_dst:
                new_dst = None
                if origin_dst is not None:
                    dst = origin_dst
                    origin_dst = None
                    # counter = 20
                    continue

        # if counter <= 0 or paths is None or len(paths) <= 1:
        #     print('score: ', env.status.score)
        #     obs = env.reset()
        #     continue

        if use_random:
            action = np.random.randint(0, 4)
            use_random = False
        else:
            dst = paths[1]
            dire = src[0] - dst[0], src[1] - dst[1]
            action = directions[dire]
        # import ipdb
        # ipdb.set_trace()
        if need_record:
            x.append(obs)
            y.append(action)
            if len(y) >= record:
                return x, y

            if len(y) % 1000 == 0:
                print(len(y))

        _, obs, done, _ = env(action)
        # counter = 20

        if done:
            print(env.status.score)
            sleep(1.5)
            break
    if not record and not no_render:
        env.render()

    env.close()


if __name__ == '__main__':

    play(no_render=False)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    # x, y = play(no_render=False)
    # with open('datasets/train.pkl', 'wb') as fout:
    #     pickle.dump({'x': x, 'y': y}, fout)

    # x, y = play(record=10000)
    # with open('datasets/test.pkl', 'wb') as fout:
    #     pickle.dump({'x': x, 'y': y}, fout)
