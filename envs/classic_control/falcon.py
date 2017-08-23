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
        self.length = 1.5
        self.dt = 0.02  # seconds between state updates
        self.groundheight = 1.0
        self.startheight = 10.0
        self.thrustmultiplier = 20.0  # thrust = thrustmultiplier * throttle
        self.theta_threshold_radians = math.pi / 3
        self.x_threshold = 10.0
        self.y_threshold = self.startheight * 2
        self.phi_threshold = 0.3

        self.shaping = 0
        self.prev_shaping = 0

        high = np.array([
            self.x_threshold * 2,
            self.y_threshold,
            self.theta_threshold_radians * 2,
            self.phi_threshold * 2,
            0.0
        ])

        low = np.array([
            -self.x_threshold * 2,
            self.groundheight,
            -self.theta_threshold_radians * 2,
            -self.phi_threshold * 2,
            1.1
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

        x, y, theta, phi, throttle = self.state

        if action == 0:
            phi = phi + 0.05
        elif action == 1:
            phi = phi - 0.05
        elif action == 2:
            throttle = throttle + 0.1
        elif action == 3:
            throttle = throttle - 0.1

        phi = np.clip(phi, -self.phi_threshold, self.phi_threshold)
        throttle = np.clip(throttle, 0.1, 1.0)

        force_mag = self.thrustmultiplier * throttle
        x = x + self.dt * self.x_dot
        y = y + self.dt * self.y_dot
        theta = theta + self.dt * self.theta_dot
        self.x_dot = self.x_dot + 1 / self.mass * math.sin(theta + phi) * force_mag * self.dt
        self.y_dot = self.y_dot + (1 / self.mass * math.cos(theta + phi) * force_mag - self.gravity) * self.dt
        self.theta_dot = self.theta_dot - (1 / self.momentofinertia) * np.sin(phi) * force_mag * self.length / 2 * self.dt
        self.state = (x, y, theta, phi, throttle)
        altitude = y - (self.groundheight + self.length / 2)
        speed = math.sqrt(self.x_dot ** 2 + self.y_dot ** 2)
        distance = math.sqrt(x ** 2 + altitude ** 2)

        # REWARDS TODO
        # -----------------------
        done = False
        reward = 0

        if self.y_dot > 0:
            reward -=1


        # game over
        if y > self.y_threshold or abs(x) > self.x_threshold or abs(theta) > self.theta_threshold_radians:
            done = True
            reward += -50
        elif altitude < 3 and abs(x) < 3 and speed < 3 and abs(theta) < 0.3:
            done = True
            reward += max(0, 200 - altitude**2 - x**2 - speed**2 - theta**2)
            print("LANDED ", reward)

        # -----------------------

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.theta_dot, self.x_dot, self.y_dot  = random.uniform(-0.3, 0.3), random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)
        self.state = [
                      random.uniform(-1.0, 1.0),  # x
                      self.startheight,  # y
                      random.uniform(-self.theta_threshold_radians / 2, self.theta_threshold_radians / 2),  # theta
                      random.uniform(-self.phi_threshold, self.phi_threshold),  # phi
                      random.uniform(0.1, 1)  # throttle
                      ]
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='rgb_array', close=False):
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
            leg_span = width_rocket * 6
            leg_clearing = width_rocket * 1.3

            self.rockettrans = rendering.Transform()

            l, r, t, b = -width_rocket / 2, width_rocket / 2, length_rocket, -length_rocket
            rocket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            rocket.add_attr(self.rockettrans)
            self.viewer.add_geom(rocket)

            leg_left = rendering.FilledPolygon(
                [(width_rocket / 2, -length_rocket),
                 (width_rocket / 2, width_rocket * 0.8 - length_rocket),
                 (leg_span / 2, - leg_clearing - length_rocket)
                 ])
            leg_left.add_attr(self.rockettrans)
            self.viewer.add_geom(leg_left)

            leg_right = rendering.FilledPolygon(
                [(-width_rocket / 2, -length_rocket),
                 (-width_rocket / 2, width_rocket * 0.8 - length_rocket),
                 (-leg_span / 2, - leg_clearing - length_rocket)
                 ])
            leg_right.add_attr(self.rockettrans)
            self.viewer.add_geom(leg_right)

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
            fire.set_color(0.9, 0.6, 0.3)
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
        ypos = x[1] * scale
        self.rockettrans.set_translation(xpos, ypos)
        self.rockettrans.set_rotation(-x[2])
        self.enginetrans.set_rotation(-x[3])
        self.firescale.set_scale(newx=1, newy=x[4])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
