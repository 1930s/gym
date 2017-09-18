import math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, distanceJointDef,
                      contactListener)
import gym
from gym import spaces
from gym.utils import seeding

# Rocket trajectory optimization is a classic topic in Optimal Control.
CONTINUOUS = False

FPS = 50
SCALE_S = 0.5  # Temporal Scaling

INITIAL_RANDOM = 1.0

START_FUEL = 250.0
START_HEIGHT = 400.0
START_SPEED = 100.0

# ROCKET
ROCKET_RATIO = 14
ROCKET_WIDTH = 3.7 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 40
ENGINE_HEIGHT = ROCKET_WIDTH * 0.5
ENGINE_WIDTH = ENGINE_HEIGHT * 0.7
MAIN_ENGINE_POWER = 10000 * SCALE_S ** 2
SIDE_ENGINE_POWER = 50 * SCALE_S ** 2
GIMBAL_THRESHOLD = 0.4
LANDER_POLY = [
    (-ROCKET_WIDTH / 2, 0),
    (+ROCKET_WIDTH / 2, 0),
    (ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
    (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT)
]
ENGINE_POLY = [
    (0, 0),
    (ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
    (-ENGINE_WIDTH / 2, -ENGINE_HEIGHT)
]

# LEGS
LEG_LENGTH = ROCKET_WIDTH * 2.5
BASE_ANGLE = -0.3
SPRING_ANGLE = 0.3
LEG_AWAY = ROCKET_WIDTH / 2


def LEG_POLY(i):
    return ([0, 0], [0, LEG_LENGTH / 25], [i * LEG_LENGTH, 0], [i * LEG_LENGTH, -LEG_LENGTH / 20],
            [i * LEG_LENGTH / 3, -LEG_LENGTH / 7])


BARGE_HEIGHT = ROCKET_WIDTH * 1
BARGE_WIDTH = BARGE_HEIGHT * 15

VIEWPORT_H = 1200
VIEWPORT_W = 900
H = 1.1 * START_HEIGHT * SCALE_S
W = VIEWPORT_W / VIEWPORT_H * H


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.water in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        else:
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

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.barge = None
        self.legs = []

        high = np.array([np.inf] * 7)
        self.observation_space = spaces.Box(-high, high)

        if CONTINUOUS:
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            self.action_space = spaces.Discrete(7)

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
        self.fuel = START_FUEL

        # terrain
        # self.terrainheigth = self.np_random.uniform(H / 20, H / 10)
        self.terrainheigth = H / 20
        # barge_pos = self.np_random.uniform(0, BARGE_WIDTH / SCALE) + BARGE_WIDTH / SCALE
        barge_pos = W / 2
        self.helipad_x1 = barge_pos - BARGE_WIDTH / 2
        self.helipad_x2 = self.helipad_x1 + BARGE_WIDTH
        self.helipad_y = self.terrainheigth + BARGE_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (W, 0), (W, self.terrainheigth), (0, self.terrainheigth)]),
                friction=0.1,
                restitution=0.0)
        )
        self.water.color1 = (35. / 255, 70. / 255, 177. / 255)

        self.barge = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(self.helipad_x1, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth + BARGE_HEIGHT),
                              (self.helipad_x1, self.terrainheigth + BARGE_HEIGHT)]),
                friction=0.4,
                restitution=0.0)
        )

        self.barge.color1 = (0.2, 0.2, 0.2)

        initial_x = np.random.uniform(W / 3, W * 2 / 3)
        initial_y = H * 0.95
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=LANDER_POLY),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)
        )
        self.lander.localCenter = (0, ROCKET_HEIGHT / 3)

        self.lander.color1 = (0.85, 0.85, 0.85)
        self.lander.linearVelocity = (
            self.np_random.uniform(-0.1 * START_SPEED * INITIAL_RANDOM, 0.1 * START_SPEED * INITIAL_RANDOM),
            self.np_random.uniform(-1.2 * START_SPEED * INITIAL_RANDOM, -0.8 * START_SPEED * INITIAL_RANDOM))
        self.lander.angularVelocity = self.np_random.uniform(-0.3 * INITIAL_RANDOM, 0.3 * INITIAL_RANDOM)

        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY, initial_y + ROCKET_WIDTH * 0.2),
                angle=(i * BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=LEG_POLY(i)),
                    density=1.0,
                    restitution=0.0,
                    friction=0.4,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY, ROCKET_WIDTH * 0.2),
                localAnchorB=(0, 0),
                enableMotor=False,
                enableLimit=True
            )
            djd = distanceJointDef(bodyA=self.lander,
                                   bodyB=leg,
                                   anchorA=(i * LEG_AWAY, ROCKET_HEIGHT / 8),
                                   anchorB=leg.fixtures[0].body.transform * (i * LEG_LENGTH, 0),
                                   collideConnected=False,
                                   frequencyHz=0.2,
                                   dampingRatio=10
                                   )
            if i == 1:
                rjd.lowerAngle = -SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = + SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs + [self.water] + [self.barge]

        if CONTINUOUS:
            return self._step([0, 0])[0]
        else:
            return self._step(6)[0]

    def _step(self, action):

        self.force_dir = 0

        if CONTINUOUS:
            if action[0] > 0.2:
                self.gimbal += 0.05
            elif action[0] < -0.2:
                self.gimbal -= 0.05
            if action[1] > 0.2:
                self.gimbal += 0.10
            elif action[1] < -0.2:
                self.gimbal -= 0.10
        else:
            if action == 0:
                self.gimbal += 0.1
            elif action == 1:
                self.gimbal -= 0.1
            elif action == 2:
                self.throttle += 0.15
            elif action == 3:
                self.throttle -= 0.15
            elif action == 4:  # left
                self.force_dir = -1
            elif action == 5:  # right
                self.force_dir = 1

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)

        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (-math.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.throttle,
                 math.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.throttle)
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        force_pos_s = self.lander.fixtures[0].body.transform * [0, ROCKET_HEIGHT * 0.9]
        force_s = (-self.force_dir * math.cos(self.lander.angle) * SIDE_ENGINE_POWER,
                   self.force_dir * math.sin(self.lander.angle) * SIDE_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse=force_s, point=force_pos_s, wake=False)

        self.world.Step(1.0 / FPS, 10, 10)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - W / 2) / H,
            (pos.y - self.helipad_y) / H,
            self.lander.angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.throttle,
            self.gimbal
        ]
        distance = np.linalg.norm(state[0:2])
        speed = np.linalg.norm(vel) / FPS
        angle = abs(state[2])

        self.fuel -= self.throttle
        if not self.force_dir == 0:
            self.fuel -= 0.5

        # REWARD -----------------------------------------------
        reward = -self.throttle / 100 - abs(self.force_dir) / 100 - 1 / 500

        if self.legs[0].joint.angle < -0.4 or self.legs[1].joint.angle > 0.4 or \
                        abs(pos.x - W / 2) > W / 2 or pos.y > H or self.fuel < 0:
            self.game_over = True

        done = False
        if self.game_over:
            done = True
            reward = -1.0
        else:
            shaping = - 1 * distance \
                      - 1 * speed \
                      - 1 * angle \
                      + 0.2 * (state[3] + state[4])
            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping
            self.prev_shaping = shaping

            if not self.lander.awake:
                print("LANDED SUCCESSFULLY")
                done = True
                reward = 1.0
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
            self.yellow = (206. / 255, 206. / 255, 2. / 255)
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

            sky = rendering.FilledPolygon([(0, 0), (0, H), (W, H), (W, 0)])
            sky.set_color(126. / 255, 150. / 255, 255. / 255)
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()

            # engine
            engine = rendering.FilledPolygon(ENGINE_POLY)
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(.4, .4, .4)
            self.viewer.add_geom(engine)

            # fire
            fire = rendering.FilledPolygon(
                [(ENGINE_WIDTH * 0.4, 0), (-ENGINE_WIDTH * 0.4, 0), (-ENGINE_WIDTH * 0.7, -ENGINE_HEIGHT * 4),
                 (0, -ENGINE_HEIGHT * 6), (ENGINE_WIDTH * 0.7, -ENGINE_HEIGHT * 4)])
            fire.set_color(249. / 255, 241. / 255, 114. / 255)
            self.firescale = rendering.Transform(scale=(1, 1))
            self.firetrans = rendering.Transform(translation=(0, -ENGINE_HEIGHT))
            fire.add_attr(self.firescale)
            fire.add_attr(self.firetrans)
            fire.add_attr(self.enginetrans)
            fire.add_attr(self.rockettrans)
            self.viewer.add_geom(fire)

            puff = rendering.FilledPolygon([(ROCKET_WIDTH / 2, ROCKET_HEIGHT * 0.9),
                                            (ROCKET_WIDTH * 3, ROCKET_HEIGHT * 0.93),
                                            (ROCKET_WIDTH * 4, ROCKET_HEIGHT * 0.9),
                                            (ROCKET_WIDTH * 3, ROCKET_HEIGHT * 0.87)])
            puff.set_color(185. / 255, 198. / 255, 255. / 255)
            self.puffscale = rendering.Transform(scale=(1, 1))
            puff.add_attr(self.puffscale)
            puff.add_attr(self.rockettrans)
            self.viewer.add_geom(puff)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                if hasattr(obj, 'color2'):
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for l in zip(self.legs, [-1, 1]):
            path = [self.lander.fixtures[0].body.transform * [l[1] * ROCKET_WIDTH / 2, ROCKET_HEIGHT / 8],
                    l[0].fixtures[0].body.transform * [l[1] * LEG_LENGTH * 0.8, 0]]
            self.viewer.draw_polyline(path, color=self.barge.color1, linewidth=1)

        self.viewer.draw_polyline([(self.helipad_x2, self.terrainheigth + BARGE_HEIGHT),
                                   (self.helipad_x1, self.terrainheigth + BARGE_HEIGHT)],
                                  color=self.yellow,
                                  linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.terrainheigth
            flagy2 = flagy1 + BARGE_HEIGHT * 1.5
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=self.barge.color1, linewidth=2)

        pos = self.lander.position
        self.rockettrans.set_translation(pos[0], pos[1])
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(newx=1, newy=self.throttle * np.random.uniform(0.9, 1.1))
        self.puffscale.set_scale(newx=self.force_dir, newy=1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
