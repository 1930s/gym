import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

logger = logging.getLogger(__name__)


class FalconEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.gravity = 9.81
        self.mass = 1.0
        self.momentofinertia = 1.0
        self.length = 1.0  # actually half the rocket's length
        self.dt = 0.02  # seconds between state updates
        self.groundheight = 1.0
        self.startheight = 8.0
        self.thrustmultiplier = 20.0  # thrust = thrustmultiplier * throttle

        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi / 4
        self.x_threshold = 10.0
        self.y_threshold = self.startheight*2
        self.phi_threshold = 0.3

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.y_threshold,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
            self.phi_threshold,
            0.1
        ])

        low = np.array([
            -self.x_threshold * 2,
            -np.finfo(np.float32).max,
            self.groundheight,
            -np.finfo(np.float32).max,
            -self.theta_threshold_radians * 2,
            -np.finfo(np.float32).max,
            -self.phi_threshold,
            1.0
        ])

        self.action_space = spaces.Discrete(4)  # gimbal(left,right), throttle(up,down)
        self.observation_space = spaces.Box(low, high)  # x, x_dot, y, y_dot, theta, theta_dot, phi, throttle

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x, x_dot, y, y_dot, theta, theta_dot, phi, throttle = self.state

        if action == 0:
            phi = -self.phi_threshold
        elif action == 1:
            phi = self.phi_threshold
        elif action == 2:
            throttle = 1
        elif action == 3:
            throttle = 0.1

        phi = np.clip(phi, -1, 1)
        throttle = np.clip(throttle, 0.1, 1.0)

        force_mag = self.thrustmultiplier * throttle

        x = x + self.dt * x_dot
        y = y + self.dt * y_dot
        theta = theta + self.dt * theta_dot
        x_dot = x_dot + 1 / self.mass * math.sin(theta + phi) * force_mag * self.dt
        y_dot = y_dot + (1 / self.mass * math.cos(theta + phi) * force_mag - self.gravity) * self.dt
        theta_dot = theta_dot - (1 / self.momentofinertia) * np.sin(phi) * force_mag * self.length / 2 * self.dt

        self.state = (x, x_dot, y, y_dot, theta, theta_dot, phi, throttle)
        done = y < self.groundheight \
            or y > self.y_threshold \
            or abs(x) > self.x_threshold \
            or abs(theta) > self.theta_threshold_radians

        if not done:
            reward = 1  # TODO
        else:
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(8,))
        self.state = [
            random.randint(-3, 3),  # x
            0,
            self.startheight + random.randint(-3, 3),  # y
            0,
            float(np.random.randint(-4, 4)) / 10,  # theta
            float(np.random.randint(-4, 4)) / 10,  # theta_dot
            float(np.random.randint(-2, 2)) / 10,  # phi
            float(np.random.randint(1, 4)) / 10  # throttle
        ]
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 1200
        screen_height = 800

        world_width = self.x_threshold * 2
        scale = screen_width / world_width

        rocket_ratio = 5
        width_rocket = 20
        length_rocket = width_rocket * rocket_ratio
        width_engine = width_rocket * 0.8
        width_fire = width_engine * 0.9

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # rocket
            l, r, t, b = -width_rocket / 2, width_rocket / 2, length_rocket, -length_rocket
            rocket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            rocket.set_color(0, 0, 0)
            self.rockettrans = rendering.Transform()
            rocket.add_attr(self.rockettrans)
            self.viewer.add_geom(rocket)

            # engine
            l, r, t_engine = -width_engine / 2, width_engine / 2, width_engine
            engine = rendering.FilledPolygon([(0, 0), (l / 3, 0), (l, -t_engine), (r, -t_engine), (r / 3, 0)])
            engine.set_color(0.4, 0.4, 0.4)
            self.enginetrans = rendering.Transform(translation=(0, -length_rocket * 1))
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            self.viewer.add_geom(engine)

            # fire
            l, r, t = -width_fire / 2, width_fire / 2, width_fire * 2
            fire = rendering.FilledPolygon([(l, 0), (r, 0), (0, -t)])
            fire.set_color(0.8, 0, 0)
            self.firescale = rendering.Transform(scale=(1, 1))
            self.firetrans = rendering.Transform(translation=(0, -t_engine))
            fire.add_attr(self.firescale)
            fire.add_attr(self.firetrans)
            fire.add_attr(self.enginetrans)
            fire.add_attr(self.rockettrans)
            self.viewer.add_geom(fire)

            # ground
            ground = rendering.FilledPolygon(
                [(0, self.groundheight * scale), (0, 0), (screen_width, 0), (screen_width, self.groundheight * scale)])
            self.firetrans = rendering.Transform(translation=(0, 0))
            ground.set_color(0, 0, 0)
            self.viewer.add_geom(ground)

        if self.state is None:
            return None

        x = self.state
        xpos = x[0] * scale + screen_width / 2.0
        ypos = x[2] * scale
        self.rockettrans.set_translation(xpos, ypos)
        self.rockettrans.set_rotation(-x[4])
        self.enginetrans.set_rotation(-x[6])
        self.firescale.set_scale(newx=1, newy=x[7])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
