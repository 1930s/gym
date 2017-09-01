import math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import seeding

# Rocket trajectory optimization is a classic topic in Optimal Control.


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

# ROCKET
ROCKET_RATIO = 10
ROCKET_WIDTH = 20
ENGINE_SIZE = ROCKET_WIDTH * 0.8
MAIN_ENGINE_POWER = 300
GIMBAL_THRESHOLD = 0.4
LANDER_POLY = [
    (ROCKET_WIDTH / 2, +ROCKET_WIDTH * ROCKET_RATIO),
    (-ROCKET_WIDTH / 2, 0),
    (+ROCKET_WIDTH / 2, 0),
    (-ROCKET_WIDTH / 2, +ROCKET_WIDTH * ROCKET_RATIO)
]
ENGINE_POLY = [
    (0, 0),
    (ENGINE_SIZE / 2, -ENGINE_SIZE / 1),
    (-ENGINE_SIZE / 2, -ENGINE_SIZE / 1)
]

# LEGS
LEG_SCALE = 1.2
BASE_ANGLE = 0.2
SPRING_ANGLE = 0.1
LEG_AWAY = ROCKET_WIDTH / 2
LEG_DOWN = 0
LEG_W, LEG_H = 4, 25
LEG_SPRING_TORQUE = 40

BARGE_HEIGHT = 25
BARGE_WIDTH = 300

VIEWPORT_W = 900
VIEWPORT_H = 1200
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.water in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True

        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class RocketLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.barge = None
        self.legs = []

        high = np.array([np.inf] * 8)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # gimbal (left/right) or throttle (up/down)
            self.action_space = spaces.Discrete(4)

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.water: return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.barge)
        self.barge = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []

    def _reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0.0
        self.gimbal = 0.0

        # terrain TODO
        self.terrainheigth = self.np_random.uniform(H / 20, H / 10)
        barge_pos = self.np_random.uniform(0, BARGE_WIDTH / SCALE) + BARGE_WIDTH / SCALE
        self.helipad_x1 = barge_pos
        self.helipad_x2 = self.helipad_x1 + BARGE_WIDTH / SCALE
        self.helipad_y = self.terrainheigth + BARGE_HEIGHT / SCALE

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (W, 0), (W, self.terrainheigth), (0, self.terrainheigth)]),
                density=0,
                friction=0.1,
                restitution=0.0)
        )
        self.water.color1 = (0.53 / 2, 0.81 / 2, 0.98 / 2)
        self.water.color2 = (0.53 / 2, 0.81 / 2, 0.98 / 2)

        self.barge = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(self.helipad_x1, self.terrainheigth), (self.helipad_x2, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth + BARGE_HEIGHT / SCALE),
                              (self.helipad_x1, self.terrainheigth + BARGE_HEIGHT / SCALE)]),
                density=0,
                friction=0.1,
                restitution=0.0)
        )

        self.barge.color1 = (0.2, 0.2, 0.2)
        self.barge.color2 = self.barge.color1

        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, H),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)
        )
        self.lander.color1 = (0.8, 0.8, 0.8)
        self.lander.color2 = (0.7, 0.7, 0.7)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-20 * INITIAL_RANDOM, -10 * INITIAL_RANDOM)
        ), True)

        legsize = ROCKET_WIDTH / SCALE * LEG_SCALE
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, H),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=([0, 0], [i * legsize / 2, -legsize], [i * 2 * legsize, -2 * legsize])),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.2, 0.2, 0.2)
            leg.color2 = (0.3, 0.3, 0.3)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=-0.3 * i  # low enough not to jump back into the sky
            )
            if i == 1:
                rjd.lowerAngle = +BASE_ANGLE - SPRING_ANGLE  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +BASE_ANGLE
            else:
                rjd.lowerAngle = -BASE_ANGLE
                rjd.upperAngle = -BASE_ANGLE + SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs + [self.water] + [self.barge]

        return self._step(np.array([0, 0]) if self.continuous else 0)[0]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        if action == 0:
            self.gimbal += 0.1
        elif action == 1:
            self.gimbal -= 0.1
        elif action == 2:
            self.throttle += 0.2
        elif action == 3:
            self.throttle -= 0.2

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.1, 1.0)
        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        ox = tip[0]
        oy = -tip[1]
        impulse_pos = (self.lander.position[0], self.lander.position[1])
        impulse = (-math.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.throttle,
                   math.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.throttle)
        self.lander.ApplyForce(impulse, impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_W / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.throttle,
            self.gimbal
        ]
        assert len(state) == 10

        # REWARD -----------------------------------------------
        reward = 0
        shaping = \
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        # REWARD -----------------------------------------------


        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

            sky = rendering.FilledPolygon([(0, 0), (0, H), (W, H), (W, 0)])
            sky.set_color(0.53, 0.81, 0.98)
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()
            # engine
            engine = rendering.FilledPolygon([(x / SCALE, y / SCALE) for x, y in ENGINE_POLY])
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(0.3, 0.3, 0.3)
            self.viewer.add_geom(engine)
            # fire
            width_fire = ENGINE_SIZE / SCALE * 0.5
            l, r, t = -width_fire / 2, width_fire / 2, width_fire * 3
            fire = rendering.FilledPolygon([(0, -t / 6), (l, -t / 3), (0, -t), (r, -t / 3)])
            fire.set_color(0.9, 0.6, 0.3)
            self.firescale = rendering.Transform(scale=(1, 1))
            self.firetrans = rendering.Transform(translation=(0, -ENGINE_SIZE / SCALE))
            fire.add_attr(self.firescale)
            fire.add_attr(self.firetrans)
            fire.add_attr(self.enginetrans)
            fire.add_attr(self.rockettrans)
            self.viewer.add_geom(fire)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)],
                                     color=(0.8, 0.8, 0))

        pos = self.lander.position
        self.rockettrans.set_translation(pos[0], pos[1])
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(newx=self.throttle, newy=self.throttle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


