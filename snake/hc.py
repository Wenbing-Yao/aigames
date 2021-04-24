# -*- coding: utf-8 -*-
from copy import deepcopy
from itertools import combinations
from os.path import expandvars
import ipdb

import numpy as np
import pygame

from snake_env import SnakeEnv
from snake_env import GRID_WIDTH_NUM
from snake_env import GRID_HEIGHT_NUM

from time import sleep
import numpy as np
from snake_env import SnakeEnv

from collections import deque
import pickle
MAX_DEPTH = 18


def show_graph(graph, flags, env, update=True, width=2, extra=None):
    # env.reset()
    if not extra:
        env.render(update=not update)
    else:
        env.render(extra, update=not update)
    for sp in graph:
        for ep in graph[sp]:
            if flags[(sp, ep)]:
                env.draw_connection(sp, ep, color=(0, 0xff, 0), width=width)

    if update:
        pygame.display.update()


def deletion(graph, env):
    flags = {}

    edges = set()

    perm = np.random.permutation(len(graph))
    for sp in graph:
        for ep in graph[sp]:
            flags[(sp, ep)] = 1
            edges.add((sp, ep))

    show_graph(graph, flags, env)
    sleep(1)
    perm = np.random.permutation(len(edges))
    edges = list(edges)
    for i in perm:
        sp, ep = edges[i]
        # for sp, ep in edges:
        if len(graph[sp]) > 2 and len(graph[ep]) > 2:
            if ep in graph[sp]:
                graph[sp].remove(ep)
                flags[(sp, ep)] = 0
            if sp in graph[ep]:
                graph[ep].remove(sp)
                flags[(ep, sp)] = 0

        if i % 10 == 0:
            show_graph(graph, flags, env)

    return graph, flags


def destroy_path(graph, total_graph, flags, start, end, w, visited, deps=1):
    if deps > MAX_DEPTH or start in visited:
        visited.add(start)
        return 0

    visited.add(start)
    edges = []
    for ep in total_graph[start]:
        # print('-' * 4 * deps, 'flags', (
        #     start,
        #     ep,
        # ), flags[(start, ep)])
        if flags[(start, ep)] == w:
            edges.append((start, ep))

    # print('-' * 4 * deps, 'start at', start, w)
    # print('-' * 4 * deps, 'found', edges)
    for edge in edges:
        ep = edge[1]
        if ep in visited:
            continue

        # print('-' * 4 * deps, ep)
        if flags[edge] == 1 and w == 1 and end == ep:
            # print('-' * 4 * deps, 'found 1', start, ep)
            return [ep]

        res = destroy_path(graph,
                           total_graph,
                           flags,
                           start=ep,
                           end=end,
                           w=1 - w,
                           visited=visited,
                           deps=deps + 1)
        if res:
            res.append(ep)
            return res
        else:
            # visited.remove(ep)
            # print('-' * 4 * deps, 'ends 0', ep)
            pass

    return []


def reflect(e, graph, flags):
    if flags[e]:
        flags[e] = 0
        graph[e[0]].remove(e[1])
    else:
        flags[e] = 1
        graph[e[0]].add(e[1])


def reflect_path(path, graph, flags):
    for i in range(len(path) - 1):
        e1 = (path[i], path[i + 1])
        e2 = (path[i + 1], path[i])
        reflect(e1, graph, flags)
        reflect(e2, graph, flags)


def get_sd(graph):
    SD = []
    for v in graph:
        if len(graph[v]) > 2:
            SD.append(v)

    return SD


def destroy(graph, total_graph, flags, env=None):

    SD = get_sd(graph)
    if not SD:
        return 0

    w = 1
    for start, end in combinations(SD, 2):
        # print(start, end)
        visited = set()
        path = destroy_path(graph, total_graph, flags, start, end, w, visited)

        show_graph(graph, flags, env)
        if path:
            path.append(start)
            # print(paths)
            reflect_path(path, graph, flags)
            break
        else:
            print('not found')

        if env:
            show_graph(graph, flags, env)

    return len(get_sd(graph))


def connecting_path(start,
                    end,
                    w,
                    f,
                    graph,
                    total_graph,
                    flags,
                    visited,
                    deps=1):
    edges = []
    visited.add(start)
    global MAX_DEPTH
    if len(visited) > 15:
        return []

    if len(visited) > MAX_DEPTH:
        return []

    for ep in total_graph[start]:
        if flags[(start, ep)] == w:
            edges.append((start, ep))

    # print('-' * deps * 2, edges)

    for edge in edges:
        sp, ep = edge
        if ep != end and ep in visited:

            # print('-' * deps * 2, edge, end, flags[edge], 'visited', visited)
            continue

        # print('-' * deps * 2, edge, end, flags[edge])
        if flags[edge] == 0 and end == edge[1]:
            # print('-' * deps, 'found: ', edge)
            return [edge[1]]

        if end == edge[1]:
            continue

        res = connecting_path(ep, end, 1 - w, 1 - f, graph, total_graph, flags,
                              visited, deps + 1)
        if not res:
            visited.remove(ep)
            continue

        res.append(ep)
        # print(res)
        return res

    return []


def get_circle(graph, origin=0):
    visited = set()
    start = origin
    while True:
        if start in visited:
            return visited

        visited.add(start)
        for v in graph[start]:
            if v in visited:
                continue

            start = v
            break


def get_list_circle(graph, origin=0):
    visited = set()
    start = origin
    seq = []
    while True:
        if start in visited:
            return seq

        visited.add(start)
        seq.append(start)
        for v in graph[start]:
            if v in visited:
                continue

            start = v
            break


def get_all_sets(graph):
    total = set(graph.keys())
    batch = []
    while total:
        v = total.pop()
        batch.append(get_circle(graph, origin=v))
        total -= batch[-1]

    batch = sorted(batch, key=lambda b: len(b))

    return batch


def get_smallest_circle(graph):
    return get_all_sets(graph)[0].pop()


def num_of_circle(graph):
    return len(get_all_sets(graph))


def sorted_vetexes(graph):
    vetexes = []
    for batch in get_all_sets(graph):
        vetexes.extend(batch)

    return vetexes


def has_one_circle(graph, origin=0):
    get_all_sets(graph)

    return len(get_circle(graph)) == len(graph)


def connector(graph, total_graph, flags, env=None):
    if has_one_circle(graph):
        print('match')
        show_graph(graph, flags, env)
        return True

    for v in sorted_vetexes(graph):
        w = 1
        visited = set()
        path = connecting_path(v, v, w, w, graph, total_graph, flags, visited)
        if path:
            path.append(v)
            path = list(reversed(path))
            n_circles = num_of_circle(graph)
            reflect_path(path, graph, flags)
            post_n_circles = num_of_circle(graph)
            if post_n_circles > n_circles:
                reflect_path(path, graph, flags)
            else:
                show_graph(graph, flags, env)
                print(post_n_circles)

            if post_n_circles == 1:
                return True

    return False


def build_graph(row, col):
    # print(row, col)
    graphs = {}
    for r in range(row):
        for c in range(0, col - 1):
            start = r * col + c
            if start not in graphs:
                graphs[start] = set()
            if start + 1 not in graphs:
                graphs[start + 1] = set()

            graphs[start + 1].add(start)
            graphs[start].add(start + 1)

    for r in range(0, row - 1):
        for c in range(col):
            start = r * col + c
            if start not in graphs:
                graphs[start] = set()
            if start + col not in graphs:
                graphs[start + col] = set()

            graphs[start].add(start + col)
            graphs[start + col].add(start)

    return graphs


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


def dfs_policy(obs, env):

    directions = {
        (-1, 0): env.right,
        (1, 0): env.left,
        (0, -1): env.down,
        (0, 1): env.up
    }
    try:
        src = np.where(obs == 2)
        src = int(src[1]), int(src[0])
        dst = np.where(obs == -1)
        dst = int(dst[0]), int(dst[1])
    except Exception:
        return None, None

    paths = bfs(obs, start=src, dst=dst)
    if paths is None or len(paths) <= 1:
        return None, None

    dst = paths[1]
    dire = src[0] - dst[0], src[1] - dst[1]
    action = directions[dire]

    return action, dst


def rel_pos(pos, rel, n_max):
    return (pos - rel) % n_max


def draw_graph():
    print(GRID_HEIGHT_NUM, GRID_WIDTH_NUM)
    # input()
    graph = build_graph(row=GRID_HEIGHT_NUM, col=GRID_WIDTH_NUM)
    total_graph = deepcopy(graph)
    env = SnakeEnv(set_life=100000, alg='HC + BFS', no_sight_disp=True)
    env.reset()
    sleep(1)

    graph, flags = deletion(graph, env)
    for sp in graph:
        for ep in graph[sp]:
            if flags[(sp, ep)]:
                # print(sp, ep)
                env.draw_connection(sp, ep, width=4)
            # env.render()

    import pygame
    pygame.display.update()
    pre_len = None
    while True:
        sd_len = destroy(graph, total_graph, flags, env=env)
        print('sd: ', sd_len)
        if pre_len is not None and pre_len == sd_len:
            global MAX_DEPTH
            print('+1')
            MAX_DEPTH += 1

        pre_len = sd_len

        show_graph(graph, flags, env)
        if not sd_len:
            break

    sleep(1)

    show_graph(graph, flags, env)
    counter = 0
    while not connector(graph, total_graph, flags, env):
        counter += 1
        print('counter: ', counter)

    sleep(1)

    for sp in graph:
        for ep in graph[sp]:
            if flags[(sp, ep)]:
                env.draw_connection(sp, ep, color=(0xff, 0xff, 0), width=4)

    import pygame
    show_graph(graph, flags, env)
    circle = get_list_circle(graph)
    print(circle)
    pos_encoder = {pos: i for i, pos in enumerate(circle)}
    # pos_decoder = {i: pos for i, pos in enumerate(circle)}
    pos_xy_decoder = {
        i: (pos % GRID_WIDTH_NUM, pos // GRID_WIDTH_NUM)
        for i, pos in enumerate(circle)
    }
    pos_xy_encoder = {(pos % GRID_WIDTH_NUM, pos // GRID_WIDTH_NUM): i
                      for i, pos in enumerate(circle)}
    obs = env.reset()
    c = 0
    while True:
        c += 1

        if len(env.status.snake_body) < 15:
            remainder = 20
        elif len(env.status.snake_body) < 30:
            remainder = 20
        elif len(env.status.snake_body) < 60:
            remainder = 30
        elif len(env.status.snake_body) < 90:
            remainder = 30
        elif len(env.status.snake_body) < 120:
            remainder = 40
        elif len(env.status.snake_body) < 150:
            remainder = 80
        elif len(env.status.snake_body) < 300:
            remainder = 100
        elif len(env.status.snake_body) < (GRID_WIDTH_NUM * GRID_HEIGHT_NUM -
                                           10):
            remainder = 30
        else:
            remainder = 5
        bfs_action, dst = dfs_policy(obs, env)
        bfs_dst_idx = 100000000
        if dst:
            bfs_dst_idx = pos_xy_encoder[dst]
        head = env.status.snake_body[0]
        head_pos, tail_pos = pos_xy_encoder[head], pos_xy_encoder[
            env.status.snake_body[-1]]
        head_idx, tail_idx = pos_xy_encoder[head], pos_xy_encoder[
            env.status.snake_body[-1]]

        hc_next_pos = pos_xy_decoder[(head_pos + 1) % len(graph)]

        directions = {
            (-1, 0): env.right,
            (1, 0): env.left,
            (0, -1): env.down,
            (0, 1): env.up
        }
        dire = head[0] - hc_next_pos[0], head[1] - hc_next_pos[1]
        print(head, hc_next_pos, dst, dst not in env.status.snake_body[:-1])
        print(head_idx, tail_idx, bfs_dst_idx)
        action = directions[dire]
        if not env.status.food_pos:
            show_graph(graph,
                       flags,
                       env,
                       update=True,
                       width=1,
                       extra='倍速: {} X'.format(remainder * 5))
            break

        food_idx = pos_xy_encoder[env.status.food_pos]
        if bfs_action:
            print(food_idx, bfs_dst_idx, head_idx, tail_idx)
            print(rel_pos(food_idx, tail_idx, len(graph)),
                  rel_pos(bfs_dst_idx, tail_idx, len(graph)),
                  rel_pos(head_idx, tail_idx, len(graph)),
                  rel_pos(tail_idx, tail_idx, len(graph)))
            if rel_pos(food_idx, tail_idx, len(graph)) >= rel_pos(
                    bfs_dst_idx, tail_idx, len(graph)) >= rel_pos(
                        head_idx, tail_idx, len(graph)) >= rel_pos(
                            tail_idx, tail_idx, len(graph)):
                action = bfs_action
                pass

        reward, obs, done, _ = env(action)
        if done:
            show_graph(graph,
                       flags,
                       env,
                       update=True,
                       width=1,
                       extra='倍速: {} X'.format(remainder * 5))
            print(done)
            break
        # env.screen.blit(env.background, (0, 0))

        if c % remainder == 0:
            show_graph(graph,
                       flags,
                       env,
                       update=True,
                       width=1,
                       extra='倍速: {} X'.format(remainder * 5))
        # env.render(blit=False)

    show_graph(graph, flags, env, update=True, width=1)
    sleep(10)
    input()


if __name__ == '__main__':
    draw_graph()
