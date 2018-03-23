#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from scipy.integrate import odeint
from mpmath import gammainc


class PhaseChangeSimulation(object):
    def __init__(
        self,
        c_h,
        c_p,
        q,
        alpha,
        beta,
        t_env,
        n_max=np.inf,
        boiling_point=34.,
        evp_heat=142 * 200,
        t_max=100,
        resolution=0.01,
        switch_off=None
    ):
        self.c_h = c_h  # ヒータの熱容量 [J/K]
        self.c_p = c_p  # パウチの熱容量 [J/K]
        self.q = q  # ヒータの発熱量　[W]
        self.alpha = alpha  # パウチ - ヒータ間熱伝達係数
        self.beta = beta  # パウチ - 外気間熱伝達係数
        self.t_env = t_env  # 外気温 [℃]
        self.t_boil = boiling_point  # 沸点 [℃]
        self.n_max = n_max  # パウチに内容する液量 [mol]
        self.evp_heat = evp_heat  # 気化熱 [J / mol]

        self.t_initial_values = [t_env, t_env]

        self.t_max = t_max
        self.resolution = resolution
        self.time = np.arange(0, t_max, resolution)

        self.switch_off = switch_off

    def solve(self):
        th, tp, nt = self.solve_heating()

        if self.switch_off is not None:
            idx_release_start = int(self.switch_off / self.resolution)
            th_, tp_, nt_ = self.solve_release(
                th[idx_release_start], tp[idx_release_start], nt[idx_release_start])

            th = np.concatenate([th[:idx_release_start], th_])
            tp = np.concatenate([tp[:idx_release_start], tp_])
            nt = np.concatenate([nt[:idx_release_start], nt_])

        self.result = {
            't [sec]': np.arange(0, len(th)) * self.resolution,
            't_h [℃]': th,
            't_p [℃]': tp,
            'n [mol]': nt,
        }

        return self

    @staticmethod
    # v[0]:Th, v[1]:Tp
    def func_state_1(v, t, c_h, c_p, q, alpha, beta, t_env):
        return [
            (q - alpha * (v[0] - v[1])) / c_h,
            (alpha * (v[0] - v[1]) - beta * (v[1] - t_env)) / c_p
        ]

    def solve_heating(self):
        th_solutions = []
        tp_solutions = []
        nt_solutions = []

        solution_state_1 = self.solve_state_1()

        if not np.any(solution_state_1 >= self.t_boil):
            # 永遠に沸点に達しない
            raise ValueError(
                'temperature of the pouch never reaches the limit')

        solution_state_1 = solution_state_1[solution_state_1[:, 1] < self.t_boil]

        th_solutions.append(solution_state_1[:, 0])
        tp_solutions.append(solution_state_1[:, 1])
        nt_solutions.append(np.array([0] * len(solution_state_1)))

        print('state 1 in heating: {}sec'.format(
            len(solution_state_1) * self.resolution))

        t_h0 = solution_state_1[-1, 0]

        solution_state_2 = self.solve_state_2(t_h0=t_h0)
        th_state_2, tp_state_2, nt_state_2 = solution_state_2

        max_reached = np.any(nt_state_2 > self.n_max)

        if max_reached:
            th_state_2 = th_state_2[nt_state_2 <= self.n_max]
            tp_state_2 = tp_state_2[nt_state_2 <= self.n_max]
            nt_state_2 = nt_state_2[nt_state_2 <= self.n_max]

        th_solutions.append(th_state_2)
        tp_solutions.append(tp_state_2)
        nt_solutions.append(nt_state_2)

        print('state 2 in heating: {}sec'.format(
            len(th_state_2) * self.resolution))

        if max_reached:
            t_h0 = th_state_2[-1]
            solution_state_3 = self.solve_state_3(t_h0)
            th_solutions.append(solution_state_3[:, 0])
            tp_solutions.append(solution_state_3[:, 1])
            nt_solutions.append(np.array([self.n_max] * len(solution_state_3)))

        return np.concatenate(th_solutions), np.concatenate(tp_solutions), np.concatenate(nt_solutions)

    def solve_release(self, t_h0, t_p0, n_0):
        assert n_0 <= self.n_max

        th_solutions = []
        tp_solutions = []
        nt_solutions = []

        if n_0 == self.n_max:
            solution_state_3 = self.solve_release_state_3(t_h0, t_p0)

            th_solutions.append(solution_state_3[:, 0])
            tp_solutions.append(solution_state_3[:, 1])
            nt_solutions.append(np.array([n_0] * len(solution_state_3)))

            print('state 3 in release: {}sec'.format(
                len(solution_state_3) * self.resolution))

            t_h0 = solution_state_3[-1, 0]
            t_p0 = self.t_boil

        if n_0 > 0:
            solution_state_2 = self.solve_release_state_2(t_h0, n_0)
            th_solutions.append(solution_state_2[0])
            tp_solutions.append(solution_state_2[1])
            nt_solutions.append(solution_state_2[2])

            print('state 2 in release: {}sec'.format(
                len(solution_state_2[0]) * self.resolution))

            t_h0 = solution_state_2[0][-1]
            t_p0 = solution_state_2[1][-1]
            n_0 = 0.

        solution_state_1 = self.solve_release_state_1(t_h0)
        th_solutions.append(solution_state_1[:, 0])
        tp_solutions.append(solution_state_1[:, 1])
        nt_solutions.append(np.array([n_0] * len(solution_state_1)))

        return np.concatenate(th_solutions), np.concatenate(tp_solutions), np.concatenate(nt_solutions)

    # Th(t), Tp(t)について数値的に解く
    def solve_state_1(self):
        solution = odeint(
            self.func_state_1,
            self.t_initial_values,
            self.time,
            args=(self.c_h, self.c_p, self.q,
                  self.alpha, self.beta, self.t_env)
        )

        return solution

    # 沸点到達後のTh(t), Tp(t), n(t)について解く
    def solve_state_2(self, t_h0):
        t_h = - (t_h0 - self.t_boil - self.q / self.alpha) * np.exp(-self.alpha / self.c_h * self.time)\
            + 2 * t_h0 - self.t_boil - self.q / self.alpha

        t_p = np.array([self.t_boil] * len(t_h))

        c_1 = t_h0 - self.t_boil - self.q / self.alpha
        c_2 = - self.beta * (self.t_boil - self.t_env) + self.q

        n_t = (
            c_2 * self.time +
            c_1 * self.c_h / self.alpha *
            (1 - np.exp(-self.alpha / self.c_h * self.time))
        ) / self.evp_heat

        return t_h, t_p, n_t

    # 全液体が蒸発した後のTh(t), Tp(t)について解く
    def solve_state_3(self, t_h0):
        solution = odeint(
            self.func_state_1,
            [t_h0, self.t_boil],
            self.time,
            args=(self.c_h, self.c_p, self.q,
                  self.alpha, self.beta, self.t_env)
        )

        return solution

    # 全液体が蒸発した状態でスイッチoffした後、Tp(t)が沸点に戻るまでのTh(t), Tp(t)について解く
    def solve_release_state_3(self, t_h0, t_p0):
        solution = odeint(
            self.func_state_1,
            [t_h0, t_p0],
            self.time,
            args=(self.c_h, self.c_p, 0, self.alpha,
                  self.beta, self.t_env)  # 発熱量を0にする
        )

        return solution[solution[:, 1] > self.t_boil]

    # Tp(t)が沸点にあり、気体が完全に液体に戻るまでのTh(t), n(t)について解く
    def solve_release_state_2(self, t_h0, n_0):
        t_h = - (t_h0 - self.t_boil) * np.exp(-self.alpha / self.c_h * self.time)\
            + 2 * t_h0 - self.t_boil

        t_p = np.array([self.t_boil] * len(t_h))

        c_1 = t_h0 - self.t_boil
        c_2 = - self.beta * (self.t_boil - self.t_env)

        n_t = (
            c_2 * self.time +
            c_1 * self.c_h / self.alpha *
            (1 - np.exp(-self.alpha / self.c_h * self.time))
        ) / self.evp_heat + n_0

        mask = n_t > 0

        return t_h[mask], t_p[mask], n_t[mask]

    # 気体が完全に液体に戻ってからのTh(t), Tp(t)について解く
    def solve_release_state_1(self, t_h0):
        solution = odeint(
            self.func_state_1,
            [t_h0, self.t_boil],
            self.time,
            args=(self.c_h, self.c_p, 0, self.alpha,
                  self.beta, self.t_env)  # 発熱量を0にする
        )

        return solution


class PhaseChangeSimulationClassical(PhaseChangeSimulation):
    """
    ヒーターからの放熱項(gamma)を導入
    """

    def __init__(
        self,
        c_h,
        c_p,
        q,
        alpha,
        beta,
        gamma,
        t_env,
        n_max=np.inf,
        boiling_point=34.,
        evp_heat=142 * 200,
        t_max=100,
        resolution=0.01,
        switch_off=None
    ):
        super().__init__(c_h, c_p, q, alpha, beta, t_env, n_max,
                         boiling_point, evp_heat, t_max, resolution, switch_off)
        self.gamma = gamma  # ヒータ・大気間の熱伝達係数

    @staticmethod
    # v[0]:Th, v[1]:Tp
    def func_state_1(v, t, c_h, c_p, q, alpha, beta, gamma, t_env):
        return [
            (q - alpha * (v[0] - v[1]) - gamma * (v[0] - t_env)) / c_h,
            (alpha * (v[0] - v[1]) - beta * (v[1] - t_env)) / c_p
        ]

    # Th(t), Tp(t)について数値的に解く
    def solve_state_1(self):
        solution = odeint(
            self.func_state_1,
            self.t_initial_values,
            self.time,
            args=(self.c_h, self.c_p, self.q, self.alpha,
                  self.beta, self.gamma, self.t_env)
        )

        return solution

    # 沸点到達後のTh(t), Tp(t), n(t)について解く
    def solve_state_2(self, t_h0):

        t_h = - (self.q - self.gamma * (self.t_boil - self.t_env)) / (self.alpha + self.gamma) * np.exp(-(self.alpha + self.gamma) / self.c_h * self.time)\
            + (self.q - self.gamma * (self.t_boil - self.t_env)) / \
            (self.alpha + self.gamma) + t_h0

        t_p = np.array([self.t_boil] * len(t_h))

        c_1 = self.alpha * (self.q - self.gamma *
                            (self.t_boil - self.t_env)) / (self.alpha + self.gamma)
        c_2 = self.alpha * self.q / (self.alpha + self.gamma) \
            - (self.alpha * self.beta + self.beta * self.gamma + self.gamma *
               self.alpha) / (self.alpha + self.gamma) * (self.t_boil - self.t_env)

        n_t = (
            c_2 * self.time +
            c_1 * self.c_h / (self.alpha + self.gamma) *
            (1 - np.exp(- (self.alpha + self.gamma) / self.c_h * self.time))
        ) / self.evp_heat

        return t_h, t_p, n_t

    # 全液体が蒸発した後のTh(t), Tp(t)について解く
    def solve_state_3(self, t_h0):
        solution = odeint(
            self.func_state_1,
            [t_h0, self.t_boil],
            self.time,
            args=(self.c_h, self.c_p, self.q, self.alpha,
                  self.beta, self.gamma, self.t_env)
        )

        return solution

    # 全液体が蒸発した状態でスイッチoffした後、Tp(t)が沸点に戻るまでのTh(t), Tp(t)について解く
    def solve_release_state_3(self, t_h0, t_p0):
        solution = odeint(
            self.func_state_1,
            [t_h0, t_p0],
            self.time,
            args=(self.c_h, self.c_p, 0, self.alpha,
                  self.beta, self.gamma, self.t_env)  # 発熱量を0にする
        )

        return solution[solution[:, 1] > self.t_boil]

    # Tp(t)が沸点にあり、気体が完全に液体に戻るまでのTh(t), n(t)について解く
    def solve_release_state_2(self, t_h0, n_0):

        t_h = self.gamma * (self.t_boil - self.t_env) / (self.alpha + self.gamma) * np.exp(-(self.alpha + self.gamma) / self.c_h * self.time)\
            - self.gamma * (self.t_boil - self.t_env) / \
            (self.alpha + self.gamma) + t_h0

        t_p = np.array([self.t_boil] * len(t_h))

        c_1 = self.alpha * (- self.gamma * (self.t_boil -
                                            self.t_env)) / (self.alpha + self.gamma)
        c_2 = - (self.alpha * self.beta + self.beta * self.gamma + self.gamma *
                 self.alpha) / (self.alpha + self.gamma) * (self.t_boil - self.t_env)

        n_t = (
            c_2 * self.time +
            c_1 * self.c_h / (self.alpha + self.gamma) *
            (1 - np.exp(- (self.alpha + self.gamma) / self.c_h * self.time))
        ) / self.evp_heat + n_0

        mask = n_t > 0

        return t_h[mask], t_p[mask], n_t[mask]

    # 気体が完全に液体に戻ってからのTh(t), Tp(t)について解く
    def solve_release_state_1(self, t_h0):
        solution = odeint(
            self.func_state_1,
            [t_h0, self.t_boil],
            self.time,
            args=(self.c_h, self.c_p, 0, self.alpha,
                  self.beta, self.gamma, self.t_env)  # 発熱量を0にする
        )

        return solution


class PhaseChangeSimulationWithLoad(PhaseChangeSimulationClassical):
    """
    一定の荷重を持ち上げる状態を想定
    """

    def __init__(
        self,
        c_h,
        c_p,
        q,
        alpha,
        beta,
        gamma,
        t_env,
        n_max=np.inf,
        boiling_point=34.,
        evp_heat=142 * 200,
        t_max=100,
        resolution=0.01,
        switch_off=None,
        load=1.,
        pouch_width=0.025,
        pouch_height=0.1
    ):
        super().__init__(c_h, c_p, q, alpha, beta, gamma, t_env, n_max,
                         boiling_point, evp_heat, t_max, resolution, switch_off)
        self.load = load  # 荷重 [kg]
        self.pouch_width = pouch_width  # パウチの縦幅 [m]
        self.pouch_height = pouch_height  # パウチの横幅 [m]
        self.R = 8.31  # 気体定数
        self.G = 9.8  # 重力加速度
        self.P_env = 1.013e5  # 大気圧

    @staticmethod
    # v[0]:Th, v[1]:Tp, v[2]:theta
    def func_state_3(v, t, c_h, c_p, q, alpha, beta, gamma, t_env, pouch_width, load, G, n_max, R, P_env, pouch_height):
        def sim_theta():
            theta = np.linspace(0.01, np.pi / 2, 500)
            # val = (2 * n_max * R * PhaseChangeSimulationWithLoad._celcius2kelbin(v[1]) / pouch_width) \
            #     * theta * np.cos(theta) / (theta - np.cos(theta) * np.sin(theta)) \
            #     - pouch_width * pouch_height * P_env * np.cos(theta) / theta
            val = (load * G / pouch_width / pouch_height * theta / np.cos(theta) + P_env) \
                * pouch_width**2 * pouch_height / 2 * (theta - np.cos(theta) * np.sin(theta)) / theta ** 2
            # target = load * G
            target = n_max * R * \
                PhaseChangeSimulationWithLoad._celcius2kelbin(v[1])

            # return theta[val < target][0]
            return theta[val < target][-1]

        def integral_load_energy(theta):
            return PhaseChangeSimulationWithLoad.integral_theta(theta)
            # return (gammainc(1, complex(0, theta)) - gammainc(1, complex(0, -theta))).imag * (-1) / 2 \
            #     + np.sin(theta)

        next_theta = sim_theta()
        ext_energy = integral_load_energy(
            next_theta) - integral_load_energy(v[2])

        return [
            (q - alpha * (v[0] - v[1]) - gamma * (v[0] - t_env)) / c_h,
            (alpha * (v[0] - v[1]) - beta * (v[1] - t_env) - ext_energy) / c_p,
            next_theta - v[2]
        ]

    def solve(self):
        th, tp, nt, thetat = self.solve_heating()

        if self.switch_off is not None:
            idx_release_start = int(self.switch_off / self.resolution)
            th_, tp_, nt_, thetat_ = self.solve_release(
                th[idx_release_start], tp[idx_release_start], nt[idx_release_start], thetat[idx_release_start])

            th = np.concatenate([th[:idx_release_start], th_])
            tp = np.concatenate([tp[:idx_release_start], tp_])
            nt = np.concatenate([nt[:idx_release_start], nt_])
            thetat = np.concatenate([thetat[:idx_release_start], thetat_])

        self.result = {
            't [sec]': np.arange(0, len(th)) * self.resolution,
            't_h [℃]': th,
            't_p [℃]': tp,
            'n [mol]': nt,
            'angle [rad]': thetat,
        }

        return self

    def solve_heating(self):
        th_solutions = []
        tp_solutions = []
        nt_solutions = []
        thetat_solutions = []

        solution_state_1 = self.solve_state_1()

        if not np.any(solution_state_1 >= self.t_boil):
            # 永遠に沸点に達しない
            raise ValueError(
                'temperature of the pouch never reaches the limit')

        solution_state_1 = solution_state_1[solution_state_1[:, 1] < self.t_boil]

        th_solutions.append(solution_state_1[:, 0])
        tp_solutions.append(solution_state_1[:, 1])
        nt_solutions.append(np.array([0] * len(solution_state_1)))
        thetat_solutions.append(np.array([0] * len(solution_state_1)))

        print('state 1 in heating: {}sec'.format(
            len(solution_state_1) * self.resolution))

        t_h0 = solution_state_1[-1, 0]

        solution_state_2 = self.solve_state_2(t_h0=t_h0)
        th_state_2, tp_state_2, nt_state_2, thetat_state_2 = solution_state_2

        max_reached = np.any(nt_state_2 > self.n_max)

        if max_reached:
            mask = nt_state_2 <= self.n_max
            th_state_2 = th_state_2[mask]
            tp_state_2 = tp_state_2[mask]
            nt_state_2 = nt_state_2[mask]
            thetat_state_2 = thetat_state_2[mask]

        th_solutions.append(th_state_2)
        tp_solutions.append(tp_state_2)
        nt_solutions.append(nt_state_2)
        thetat_solutions.append(thetat_state_2)

        print('state 2 in heating: {}sec'.format(
            len(th_state_2) * self.resolution))

        if max_reached:
            t_h0 = th_state_2[-1]
            theta_0 = thetat_state_2[-1]
            solution_state_3 = self.solve_state_3(t_h0, theta_0)
            th_solutions.append(solution_state_3[:, 0])
            tp_solutions.append(solution_state_3[:, 1])
            thetat_solutions.append(solution_state_3[:, 2])
            nt_solutions.append(np.array([self.n_max] * len(solution_state_3)))

        return np.concatenate(th_solutions), np.concatenate(tp_solutions), np.concatenate(nt_solutions), np.concatenate(thetat_solutions)

    def solve_release(self, t_h0, t_p0, n_0, theta_0):
        assert n_0 <= self.n_max

        th_solutions = []
        tp_solutions = []
        nt_solutions = []
        thetat_solutions = []

        if n_0 == self.n_max:
            solution_state_3 = self.solve_release_state_3(t_h0, t_p0, theta_0)

            th_solutions.append(solution_state_3[:, 0])
            tp_solutions.append(solution_state_3[:, 1])
            thetat_solutions.append(solution_state_3[:, 2])
            nt_solutions.append(np.array([n_0] * len(solution_state_3)))

            print('state 3 in release: {}sec'.format(
                len(solution_state_3) * self.resolution))

            t_h0 = solution_state_3[-1, 0]
            t_p0 = self.t_boil
            theta_0 = solution_state_3[-1, 2]

        if n_0 > 0:
            solution_state_2 = self.solve_release_state_2(t_h0, n_0, theta_0)
            th_state_2, tp_state_2, nt_state_2, thetat_state_2 = solution_state_2

            n_min = nt_state_2.min()
            completely_back = n_min < 1e-8
            if completely_back:
                mask = solution_state_2[2] > n_min
                th_state_2 = th_state_2[mask]
                tp_state_2 = tp_state_2[mask]
                nt_state_2 = nt_state_2[mask]
                thetat_state_2 = thetat_state_2[mask]

            th_solutions.append(th_state_2)
            tp_solutions.append(tp_state_2)
            nt_solutions.append(nt_state_2)
            thetat_solutions.append(thetat_state_2)

            print('state 2 in release: {}sec'.format(
                len(th_state_2) * self.resolution))

            t_h0 = th_state_2[-1]
            t_p0 = tp_state_2[-1]
            n_0 = 0.

        solution_state_1 = self.solve_release_state_1(t_h0)
        th_solutions.append(solution_state_1[:, 0])
        tp_solutions.append(solution_state_1[:, 1])
        nt_solutions.append(np.array([n_0] * len(solution_state_1)))
        thetat_solutions.append(np.array([0] * len(solution_state_1)))

        return np.concatenate(th_solutions), np.concatenate(tp_solutions), np.concatenate(nt_solutions), np.concatenate(thetat_solutions)

    # 沸点到達後のTh(t), Tp(t), n(t)について解く
    def solve_state_2(self, t_h0):
        t_h = - (self.q - self.gamma * (self.t_boil - self.t_env)) / (self.alpha + self.gamma) * np.exp(-(self.alpha + self.gamma) / self.c_h * self.time)\
            + (self.q - self.gamma * (self.t_boil - self.t_env)) / \
            (self.alpha + self.gamma) + t_h0

        t_p = np.array([self.t_boil] * len(t_h))

        n_t, theta_t = self.estimate_n_evaporating()

        return t_h, t_p, n_t, theta_t

    # 全液体が蒸発した後のTh(t), Tp(t)について解く
    def solve_state_3(self, t_h0, theta_0):
        solution = odeint(
            self.func_state_3,
            [t_h0, self.t_boil, theta_0],
            self.time,
            args=(self.c_h, self.c_p, self.q, self.alpha,
                  self.beta, self.gamma, self.t_env,
                  self.pouch_width, self.load, self.G, self.n_max, self.R, self.P_env, self.pouch_height)
        )

        return solution

    def solve_release_state_3(self, t_h0, t_p0, theta_0):
        solution = odeint(
            self.func_state_3,
            [t_h0, t_p0, theta_0],
            self.time,
            args=(self.c_h, self.c_p, 0, self.alpha,
                  self.beta, self.gamma, self.t_env,
                  self.pouch_width, self.load, self.G, self.n_max, self.R, self.P_env, self.pouch_height)
        )

        return solution[solution[:, 1] > self.t_boil]

    def solve_release_state_2(self, t_h0, n_0, theta_0):
        q = 0
        t_h = - (q - self.gamma * (self.t_boil - self.t_env)) \
            / (self.alpha + self.gamma) * np.exp(-(self.alpha + self.gamma) / self.c_h * self.time) \
            + (q - self.gamma * (self.t_boil - self.t_env)) / (self.alpha + self.gamma) \
            + t_h0

        t_p = np.array([self.t_boil] * len(t_h))

        n_t, theta_t = self.estimate_n_liquefying(n_0, theta_0)

        return t_h, t_p, n_t, theta_t

    def estimate_n_evaporating(self):
        theta = np.linspace(0.01, np.pi / 2., 500)
        coef_theta = self._calc_theta_coefficients(theta)
        coef_t = self._calc_time_coefficients()

        theta_over_t = theta[self.get_closest_idx(coef_theta, coef_t)]
        # [theta[self.get_closest_idx(coef_theta, v)] for v in coef_t]
        n_over_t = (
            self.pouch_width * self.load * self.G
            * (theta_over_t - np.cos(theta_over_t) * np.sin(theta_over_t))
            / (theta_over_t * np.cos(theta_over_t))
            + self.pouch_width ** 2 * self.pouch_height * self.P_env
            * (theta_over_t - np.cos(theta_over_t) * np.sin(theta_over_t))
            / theta_over_t ** 2
        ) / (2 * self.R * self._celcius2kelbin(self.t_boil))

        return n_over_t, theta_over_t

    def estimate_n_liquefying(self, n_0, theta_0):

        theta = np.linspace(0.01, np.pi / 2., 500)
        coef_theta = self._calc_theta_coefficients(theta)
        coef_t = self._calc_time_coefficients(
            liquefy=True, theta_0=theta_0)

        theta_over_t = theta[self.get_closest_idx(coef_theta, coef_t)]
        # [theta[self.get_closest_idx(coef_theta, v)] for v in coef_t]

        print(theta_over_t)

        n_over_t = (
            self.pouch_width * self.load * self.G
            * (theta_over_t - np.cos(theta_over_t) * np.sin(theta_over_t))
            / (theta_over_t * np.cos(theta_over_t))
            + self.pouch_width ** 2 * self.pouch_height * self.P_env
            * (theta_over_t - np.cos(theta_over_t) * np.sin(theta_over_t))
            / theta_over_t ** 2
        ) / (2 * self.R * self._celcius2kelbin(self.t_boil))

        return n_over_t, theta_over_t

    def _calc_theta_coefficients(self, theta):
        # def F(theta):
        #     return \
        #         (
        #             (gammainc(1, complex(0, theta)) -
        #              gammainc(1, complex(0, -theta))).imag
        #             * (-1) / 2
        #             + np.sin(theta)
        #         ) / self.evp_heat * self.pouch_width * self.load * self.G \
        #         + (theta - np.cos(theta) * np.sin(theta)) / (theta * np.cos(theta)) \
        #         / (self._celcius2kelbin(self.t_boil) * 2 * self.R) * self.pouch_width * self.load * self.G \
        #         + (theta - np.cos(theta) * np.sin(theta)) / theta**2 \
        #         / (self._celcius2kelbin(self.t_boil) * 2 * self.R) * self.pouch_width**2 * self.pouch_height * self.P_env

        # vfunc = np.vectorize(F)

        # if isinstance(theta, np.ndarray):
        #     return vfunc(theta)

        # return F(theta)

        return self.integral_theta(theta) / self.evp_heat * self.pouch_width * self.load * self.G \
            + (theta - np.cos(theta) * np.sin(theta)) / (theta * np.cos(theta)) \
            / (self._celcius2kelbin(self.t_boil) * 2 * self.R) * self.pouch_width * self.load * self.G \
            + (theta - np.cos(theta) * np.sin(theta)) / theta**2 \
            / (self._celcius2kelbin(self.t_boil) * 2 * self.R) * self.pouch_width**2 * self.pouch_height * self.P_env

    def _calc_time_coefficients(self, liquefy=False, theta_0=None):
        q = self.q
        initial_heat = 0

        if liquefy:
            q = 0
            assert theta_0 is not None
            initial_heat = self._calc_theta_coefficients(theta_0)

        return (
            (
                self.alpha / (self.alpha + self.gamma) * q
                - (self.alpha * self.beta + self.beta *
                   self.gamma + self.gamma * self.alpha)
                / (self.alpha + self.gamma) * (self.t_boil - self.t_env)
            ) * self.time
            - self.c_h * self.alpha / (self.alpha + self.gamma)
            * (q - self.gamma * (self.t_boil - self.t_env))
            * (1. - np.exp(-(self.alpha + self.gamma) / self.c_h * self.time))
        ) / self.evp_heat + initial_heat

    @staticmethod
    def _celcius2kelbin(temperature):
        return temperature + 273.1

    @staticmethod
    def get_closest_idx(arr1, arr2):
        deviance = np.abs(arr2[np.newaxis, :] - arr1[:, np.newaxis])
        return deviance.argmin(axis=0)

    # 例の積分
    @staticmethod
    def integral_theta(theta):
        if isinstance(theta, float) and theta == 0:
            return 0

        return 1. - np.sin(theta) / theta


class LiquidPouchSimulation(object):
    def __init__(
        self,
        initial_width,
        height,
        c_h,
        c_p,
        q,
        alpha,
        beta,
        t_env,
        n_max=np.inf,
        boiling_point=34.,
        evp_heat=142 * 200,
        switch_off=None,
        Simulator=PhaseChangeSimulation,
        **kwargs
    ):
        self.R = 8.31
        self.l_0 = initial_width
        self.D = height
        self.simulator = Simulator(
            c_h=c_h,
            c_p=c_p,
            q=q,
            alpha=alpha,
            beta=beta,
            t_env=t_env,
            n_max=n_max,
            boiling_point=boiling_point,
            evp_heat=evp_heat,
            switch_off=switch_off,
            **kwargs
        ).solve()

    def get_volume(self, theta):
        self._validate_angle(theta)

        return self.l_0 ** 2 * self.D / 2 * (theta - np.cos(theta) * np.sin(theta)) / theta ** 2

    def get_tension(self, theta, t):
        t = self._take_closest_t(t)
        simulation_result = self.get_result().loc[t]
        t_p = simulation_result.t_p
        n = simulation_result.n

        return self._calc_tension(theta, n, t_p)

    def get_moment(self, theta, t):
        t = self._take_closest_t(t)
        simulation_result = self.get_result().loc[t]
        t_p = simulation_result.t_p
        n = simulation_result.n

        return self._calc_moment(theta, n, t_p)

    def get_result(self):
        return pd.DataFrame(self.simulator.result).set_index('t [sec]')

    def _calc_tension(self, theta, n, t_p):
        self._validate_angle(theta)
        theoretical_value = 2 * n * self.R * self._celcius2kelbin(t_p) / self.l_0 \
            * (theta * np.cos(theta)) / (theta - np.cos(theta) * np.sin(theta)) \
            - 1.013e5 * self.l_0 * self.D * np.cos(theta) / theta

        return max(theoretical_value, 0)

    def _calc_moment(self, theta, n, t_p):
        self._validate_angle(theta)
        theoretical_value = n * self.R * self._celcius2kelbin(t_p) \
            * (np.cos(theta) * (np.sin(theta) - theta * np.cos(theta))) \
            / (theta * (theta - np.cos(theta) * np.sin(theta))) \
            - 1.013e5 * self.l_0 ** 2 * self.D \
            * np.cos(theta) * (np.sin(theta) - theta * np.cos(theta)) / 2 / theta ** 3

        return max(theoretical_value, 0)

    @staticmethod
    def _celcius2kelbin(temperature):
        return temperature + 273.1

    def _take_closest_t(self, time):
        return min(self.get_result().index, key=lambda x: abs(x - time))

    @staticmethod
    def _validate_angle(theta):
        assert theta > 0 and theta <= np.pi / 2


if __name__ == '__main__':
    Ch = 1.5  # 2
    Cp = 3  # 4
    Q = 5.

    ALPHA = .6
    BETA = .1
    GAMMA = .05

    Ta = 22.5

    LOAD = 1.

    sim = LiquidPouchSimulation(
        initial_width=0.025,
        height=0.1,
        c_h=Ch,
        c_p=Cp,
        q=Q,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        t_env=Ta,
        n_max=1.33e-3,
        switch_off=40,
        load=LOAD,
        Simulator=PhaseChangeSimulationWithLoad
    )

    df_sim_ = sim.get_result()
    df_sim_.to_csv('sim_result_load1.csv')

    # pd.concat(data_frames).to_csv('sim_result.csv')

    # data_frames = []

    # for load in [0.05, 0.5, 1., 2., 3., 4., ]:
    #     sim = LiquidPouchSimulation(
    #         initial_width=0.025,
    #         height=0.1,
    #         c_h=Ch,
    #         c_p=Cp,
    #         q=Q,
    #         alpha=ALPHA,
    #         beta=BETA,
    #         gamma=GAMMA,
    #         t_env=Ta,
    #         n_max=1.33e-3,
    #         switch_off=40,
    #         load=load,
    #         Simulator=PhaseChangeSimulationWithLoad
    #     )

    #     df_sim_ = sim.get_result()
    #     df_sim_['load'] = load
    #     data_frames.append(df_sim_)

    # pd.concat(data_frames).to_csv('sim_result.csv')
