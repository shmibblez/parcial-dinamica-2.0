from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import pi

# angulos en radianes
# longitud en m


class Bicycle:
    """fuck"""

    def __init__(self):
        self.r = 0.25
        # anulos
        self.q1 = 0  # bike yaw
        self.q2 = 0  # bike pitch
        self.q3 = 0  # bike roll
        self.q4 = 0  # back wheel angle
        self.q5 = pi / 8  # fork pitch angle en rango [0, -pi/4]
        self.q6 = 0  # fork yaw
        self.q7 = 0  # front wheel angle
        # puntos (en marco B)
        self.a = np.array([0, 0, 0])
        self.b = np.array([0, 0, self.r])
        self.c = np.array([3 * self.r, 0, 1.5 * self.r])
        self.d = np.array([0, 0, 1.5 * self.r])

    def b_B(self):
        """el punto b en el marco B"""
        return np.array(self.b)

    def c_B(self):
        """el punto c en el marco B"""
        return np.array(self.b + 0)

    def b_C(self):
        """el punto b en el marco C"""
        # llevar punto a C (aplicar pitch)
        # TEST: if pitch increases, x coordinate should increase
        r = R.from_quat([0, 1, 0, -self.q2])
        rotated = r.apply(self.b_B())
        return np.array(rotated)

    def b_D(self):
        """el punto b en el marco D"""
        # llevar punto a D (aplicar roll)
        # TEST: if pitch increases, x coordinate should increase
        r = R.from_quat([1, 0, 0, -self.q3])
        rotated = r.apply(self.b_C())
        return np.array(rotated)

    def d_B(self):
        """el punto d en el marco B"""

    def adjust_pitch(self):
        """adjusts pitch so front wheel touches ground"""
        # vector between points b and d
        b_D = self.b_D()


b = Bicycle()
