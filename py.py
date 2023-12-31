from collections import deque
from typing import TypedDict
from scipy.spatial.transform import Rotation as R
import numpy as np
import numpy.linalg as npl
from numpy import pi
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, callback

# pio.renderers.default = "browser"

# angulos en radianes
# longitud en m


class Subspace:
    """
    un espacio que define su desplazamiento (x, y, z, ax, ay, az)
    dentro del espacio en el que esta (parent_space)
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        ax: float,
        ay: float,
        az: float,
        parent_space: "Subspace|None",
        nombre: str,
    ):
        self.parent_space = parent_space
        self.x = x
        self.y = y
        self.z = z
        self.ax = ax
        self.ay = ay
        self.az = az
        self.nombre = nombre

    @property
    def origin(self):
        """retorna origen subespacio"""
        return Punto(0, 0, 0, self, f"origen {self.nombre}")

    def rotacion_para_llevar_a_parent_space(
        self,
        target: "Subspace",
        _ax=0,
        _ay=0,
        _az=0,
        debug=False,
    ) -> R:
        """rotacion para cambio de base de actual a parent"""
        # se restan angulos de subespacios para "invertir" rotaciones
        # necesarias para quedar alineado con subespacio ancestral
        if target == self.parent_space:
            _ax += -self.ax
            _ay += -self.ay
            _az += -self.az
            return R.from_euler("xyz", [_ax, _ay, _az])
        _ax += -self.ax
        _ay += -self.ay
        _az += -self.az
        return self.parent_space.rotacion_para_llevar_a_parent_space(
            target, _ax=_ax, _ay=_ay, _az=_az, debug=debug)

    def llevar_a_parent_space(self,
                              p: "Punto",
                              target: "Subspace",
                              debug=False) -> "Punto":
        """lleva punto a parent_space s"""
        if (target == self):
            return p
        if target == self.parent_space:
            # si llegamos, retornar punto
            # print("target alcanzado")
            return Subspace.__llevar_a_parent_space__(p, self, debug=debug)
        # si no hemos llegado, pasar punto a parent,
        # y volver a intentar
        vec = Subspace.__llevar_a_parent_space__(p, self, debug=debug)
        vec = self.parent_space.llevar_a_parent_space(vec, target, debug=debug)
        return Punto.from_list(vec, target)

    @staticmethod
    def __llevar_a_parent_space__(p: "Punto",
                                  space: "Subspace",
                                  debug=False) -> "Punto":
        """
        deshace transformaciones de su espacio,
        lleva p a space.parent_space
        """
        # rotar
        vec = p
        if debug: print(f"antes: {vec}")
        r = R.from_euler(
            "xyz",
            [-space.ax, -space.ay, -space.az],
            degrees=False,
        )
        vec = r.apply(vec)
        if debug: print(f"*luego rotar {np.round(-space.az,2)}: {vec}")
        # luego trasladar
        vec = np.add(vec, [space.x, space.y, space.z])
        if debug: print(f"**luego trasladar: {vec}")
        return Punto.from_list(vec, space.parent_space)

    def llevar_a_subspace(self, p: "Punto", target: "Subspace"):
        """lleva punto a subespacio target"""
        raise NotImplementedError("no esta listo todavia")

    @staticmethod
    def __llevar_a_subspace__(p: "Punto", space: "Subspace"):
        """
        aplica transformaciones de espacio space a p,
        lleva p a space
        """
        # print("\n#llevar_a_subspace():")
        # print(
        #     f"parent (nombre: {space.nombre}) points (x,y,z): ({space.x},{space.y},{space.z})"
        # )
        # trasladar
        vec = p
        # print(f"antes: {vec}")
        vec = np.add(vec, [space.x, space.y, space.z])
        # print(f"*luego trasladar: {vec}")
        # luego rotar
        r = R.from_euler(
            "xyz",
            [-space.ax, -space.ay, -space.az],
            degrees=False,
        )
        vec = r.apply(vec)
        # print(f"**luego rotar: {vec}")
        return vec


class Punto(list[float, float, float]):
    """punto en espacio s"""

    @staticmethod
    def from_list(p: list[float, float, float],
                  parent_space: Subspace | None = None,
                  nombre: str | None = None):
        """convert list to Punto"""
        return Punto(p[0], p[1], p[2], parent_space, nombre)

    @property
    def x(self):
        """x"""
        return self[0]

    @property
    def y(self):
        """y"""
        return self[1]

    @property
    def z(self):
        """z"""
        return self[2]

    @x.setter
    def x(self, val: float):
        self[0] = val

    @y.setter
    def y(self, val: float):
        self[1] = val

    @z.setter
    def z(self, val: float):
        self[2] = val

    def __init__(self, x: float, y: float, z: float, parent_space: Subspace,
                 nombre: str):
        super().__init__([x, y, z])
        self.parent_space = parent_space
        self.nombre = nombre

    def llevar_a_parent_space(self, target: Subspace, debug=False) -> "Punto":
        """lleva punto al espacio target"""
        # print(f"target: {target}")
        nuevo_p = self.parent_space.llevar_a_parent_space(p=self,
                                                          target=target,
                                                          debug=debug)
        return Punto(nuevo_p.x,
                     nuevo_p.y,
                     nuevo_p.z,
                     target,
                     nombre=self.nombre)

    def project_onto_plane(self,
                           n: "Punto",
                           parent_space: Subspace | None,
                           nombre: str | None = None) -> "Punto":
        """project self onto plane given by normal vector n"""
        p = self - (self.dot(n) / npl.norm(n)**2) * n
        return Punto.from_list(p, parent_space=parent_space, nombre=nombre)

    def angulo_entre(self, other: "Punto") -> float:
        """returns angle between vectors"""
        return np.arccos(self.dot(other) / (npl.norm(self) * npl.norm(other)))

    def __add__(self, other: "Punto") -> "Punto":
        a = np.add(self, other)
        return Punto.from_list(a, parent_space=self.parent_space, nombre=None)

    def __sub__(self, other: "Punto") -> "Punto":
        s = np.subtract(self, other)
        return Punto.from_list(s, parent_space=self.parent_space, nombre=None)

    def __neg__(self):
        return Punto(self.x, self.y, self.z, self.parent_space, self.nombre)

    def dot(self, arr: "Punto") -> float:
        """dot product between vectors"""
        return np.dot(self, arr)

    def cross(self,
              arr: "Punto",
              parent_space: Subspace = None,
              nombre: str = None) -> "Punto":
        """cross product"""
        return Punto.from_list(np.cross(self, arr), parent_space, nombre)

    def __mul__(self, val: float) -> "Punto":
        m = np.multiply(self, val)
        return Punto.from_list(m,
                               parent_space=self.parent_space,
                               nombre=self.nombre)

    def __truediv__(self, val: float) -> "Punto":
        d = np.divide(self, val)
        return Punto.from_list(d,
                               parent_space=self.parent_space,
                               nombre=self.nombre)

    def unitario(self):
        return self / npl.norm(self)


class MatrizDeInercia(list[list[float]]):
    """
    Matriz de inercia
    aplica para origen de parent_space
    """
    # self[row][col]

    @property
    def a11(self):
        """a11"""
        return self[0][0]

    @property
    def a12(self):
        """a12"""
        return self[0][1]

    @property
    def a13(self):
        """a13"""
        return self[0][2]

    @property
    def a21(self):
        """a21"""
        return self[1][0]

    @property
    def a22(self):
        """a22"""
        return self[1][1]

    @property
    def a23(self):
        """a23"""
        return self[1][2]

    @property
    def a31(self):
        """a31"""
        return self[2][0]

    @property
    def a32(self):
        """a32"""
        return self[2][1]

    @property
    def a33(self):
        """a33"""
        return self[2][2]

    @a11.setter
    def a11(self, val: float):
        """a11"""
        self[0][0] = val

    @a12.setter
    def a12(self, val: float):
        """a12"""
        self[0][1] = val

    @a13.setter
    def a13(self, val: float):
        """a13"""
        self[0][2] = val

    @a21.setter
    def a21(self, val: float):
        """a21"""
        self[1][0] = val

    @a22.setter
    def a22(self, val: float):
        """a22"""
        self[1][1] = val

    @a23.setter
    def a23(self, val: float):
        """a23"""
        self[1][2] = val

    @a31.setter
    def a31(self, val: float):
        """a31"""
        self[2][0] = val

    @a32.setter
    def a32(self, val: float):
        """a32"""
        self[2][1] = val

    @a33.setter
    def a33(self, val: float):
        """a33"""
        self[2][2] = val

    def __init__(self, matrix: list[list[float]], parent_space: Subspace,
                 masa: float, nombre: str):
        super().__init__(matrix)
        self.space = parent_space
        self.masa = masa
        self.nombre = nombre

    def __add__(self, other: "MatrizDeInercia") -> "MatrizDeInercia":
        """sumar matrices de inercia"""
        return MatrizDeInercia(np.add(self, other), self.space,
                               self.masa + other.masa,
                               self.nombre + other.nombre)

    def __sub__(self, other: "MatrizDeInercia") -> "MatrizDeInercia":
        """restar matrices de inercia"""
        return MatrizDeInercia(np.subtract(self, other), self.space,
                               self.masa - other.masa,
                               self.nombre + other.nombre)

    def llevar_a_origen_de_parent_space(self,
                                        target: Subspace,
                                        debug: bool = False
                                        ) -> "MatrizDeInercia":
        """llevar a parent_space"""
        # primero cambio de base
        # matriz de rotacion
        M_r: list[
            list[float]] = self.space.rotacion_para_llevar_a_parent_space(
                target, debug=debug).as_matrix()
        # cambio de base, M_r * I * inv(M_r)
        basado = MatrizDeInercia(
            np.matmul(np.matmul(M_r, self), np.transpose(M_r)),
            self.space,
            self.masa,
            self.nombre,
        )

        # luego ejes paralelos para trasladar matriz de inercia
        # r va de centro de masa a punto especifico
        r = self.space.origin.llevar_a_parent_space(target)
        r_x = r.dot(Punto(1, 0, 0, target, "x"))
        r_y = r.dot(Punto(0, 1, 0, target, "y"))
        r_z = r.dot(Punto(0, 0, 1, target, "z"))
        # aplicar ecuaciones ejes paralelos para 3D
        a11 = basado.a11 + basado.masa * (r_y**2 + r_z**2)
        a22 = basado.a22 + basado.masa * (r_x**2 + r_z**2)
        a33 = basado.a33 + basado.masa * (r_x**2 + r_y**2)
        a12 = a21 = basado.a12 - basado.masa * r_x * r_y
        a13 = a31 = basado.a13 - basado.masa * r_x * r_z
        a23 = a32 = basado.a23 - basado.masa * r_y * r_z
        nuevo = MatrizDeInercia(
            [
                [a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33],
            ],
            target,
            self.masa,
            f"trasladado de { self.space.nombre } a {target.nombre}",
        )
        return nuevo
        # super().__init__([
        #     [],
        #     [],
        #     [],
        # ])

        # if debug: print(f"*luego rotar {np.round(-space.az,2)}: {vec}")
        # # luego trasladar
        # vec = np.add(vec, [space.x, space.y, space.z])
        # if debug: print(f"**luego trasladar: {vec}")
        # return Punto.from_list(vec, space.parent_space)


class Bicycle:
    """fuck"""

    def __init__(self, q1, q2, q3, q4, q5, q6, q7, d_q1, d_q4, dt, r):
        self.r = r
        self.h_manuvrio = 2 * self.r
        # angulos
        self.q1 = q1  # bike yaw
        self.q2 = q2  # bike roll
        self.q3 = q3  # bike pitch
        self.q4 = q4  # back wheel angle
        self.q5 = q5  # fork pitch angle en rango [0, -pi/4]
        self.q6 = q6  # fork yaw
        self.q7 = q7  # front wheel angle
        # velocidades angulares
        self.d_q1 = d_q1  # (parcial 3) velocidad de rotacion en vista superior,
        self.d_q4 = d_q4  # velocidad angular rueda trasera
        # variables
        self.t = 0  # tiempo actual
        self.dt = dt  # intervalo de tiempo
        # listas de datos
        # PARCIAL 1
        self.n_datos = 82  # para que sean 40 luego de doble derivada
        self.tiempo = deque(maxlen=self.n_datos)
        # 1 - posicion x, y, z de punto de interes
        self.punto_de_interes_x = deque(maxlen=self.n_datos)
        self.punto_de_interes_y = deque(maxlen=self.n_datos)
        self.punto_de_interes_z = deque(maxlen=self.n_datos)
        # 2 - velocidad de x, y, z de punto de interes
        self.punto_de_interes_dx = []
        self.punto_de_interes_dy = []
        self.punto_de_interes_dz = []
        # 2.1 - aceleracion de x, y, z de punto de interes
        self.punto_de_interes_ddx = []
        self.punto_de_interes_ddy = []
        self.punto_de_interes_ddz = []
        # 3 - velocidad angular rueda trasera y delantera
        self.omega_rueda_delantera = deque(maxlen=self.n_datos)
        self.omega_rueda_trasera = deque(maxlen=self.n_datos)
        # 4 - aceleracion angular rueda trasera y delantera
        self.alpha_rueda_delantera = deque(maxlen=self.n_datos)
        self.alpha_rueda_trasera = deque(maxlen=self.n_datos)
        # 5 - velocidad angular de interes
        self.omega_de_interes = deque(maxlen=self.n_datos)
        # 6 - aceleracion angular de interes
        self.alpha_de_interes = []
        # PARCIAL 2
        # 1 - centro de masa
        self.centro_masa_x = deque(maxlen=self.n_datos)
        self.centro_masa_y = deque(maxlen=self.n_datos)
        self.centro_masa_z = deque(maxlen=self.n_datos)
        # 2 - escalares de inercia
        self.matriz_inercia_I = deque(maxlen=self.n_datos)
        self.matriz_inercia_J = deque(maxlen=self.n_datos)
        self.matriz_inercia_K = deque(maxlen=self.n_datos)
        self.matriz_inercia_L = deque(maxlen=self.n_datos)
        # PARICAL 3
        # posicion centro de masa
        self.pos_CM = deque(maxlen=self.n_datos)
        self.pos_I = deque(maxlen=self.n_datos)
        self.pos_J = deque(maxlen=self.n_datos)
        self.pos_K = deque(maxlen=self.n_datos)
        self.pos_L = deque(maxlen=self.n_datos)
        self.pos_CM_base = deque(maxlen=self.n_datos)
        # velocidad centros de masa
        self.vel_CM = []
        self.vel_I = []
        self.vel_J = []
        self.vel_K = []
        self.vel_L = []
        self.vel_CM_base = []
        # aceleracion centros de masa
        self.acc_CM = []
        self.acc_I = []
        self.acc_J = []
        self.acc_K = []
        self.acc_L = []
        self.acc_CM_base = []
        # torque
        self.torque_total = deque(maxlen=self.n_datos)
        # marcos
        self.A = Subspace(
            x=0,  # piso es estatico
            y=0,
            z=0,
            ax=0,
            ay=0,
            az=0,
            parent_space=None,
            nombre="piso")
        # marco parcial 3, alineado con el piso para que B pueda establecer z
        # base unicamente sirve para parcial 3, para 2 y 3 es igual a B, es
        # como si no existiera
        self.base = Subspace(x=0,
                             y=0,
                             z=0,
                             ax=0,
                             ay=0,
                             az=0,
                             parent_space=self.A,
                             nombre="base")
        self.B = Subspace(
            x=0,
            y=0,
            z=0,
            ax=0,
            ay=0,
            az=self.q1,  # yaw bici
            parent_space=self.base,
            nombre="B - bici con yaw")
        self.C = Subspace(
            x=0,
            y=0,
            z=0,
            ax=self.q2,  # roll bici
            ay=0,  # pitch bici
            az=0,
            parent_space=self.B,
            nombre="C - bici con pitch")
        self.D = Subspace(
            x=0,
            y=0,
            z=0,
            ax=0,
            ay=self.q3,  # pitch bici
            az=0,
            parent_space=self.C,
            nombre="D - bici con roll")
        self.E = Subspace(
            x=0,
            y=0,
            z=self.r,
            ax=0,
            ay=self.q4,  # pitch rueda trasera
            az=0,
            parent_space=self.D,
            nombre="E - rueda trasera")
        self.F = Subspace(
            x=3 * self.r,
            y=0,
            z=self.r + self.h_manuvrio,
            ax=0,
            ay=self.q5,  # pitch marco
            az=0,
            parent_space=self.D,
            nombre="F - manubrio con pitch")
        self.G = Subspace(
            x=0,
            y=0,
            z=0,
            ax=0,
            ay=0,
            az=self.q6,  # yaw manuvrio
            parent_space=self.F,
            nombre="G - manubrio con yaw")
        self.H = Subspace(
            x=0,
            y=0,
            z=-(self.h_manuvrio) * np.cos(self.q5),
            ax=0,
            ay=self.q7,  # pitch rueda delantera
            az=0,
            parent_space=self.G,
            nombre="H - rueda delantera")
        # puntos
        # a - centro rueda trasera
        self.a = Punto(0, 0, 0, self.B, "a - contacto piso rueda trasera")
        self.b = Punto(0, 0, 0, self.E,
                       "b - centro y centro de masa rueda trasera")
        self.c = Punto(0, 0, 0, self.G, "c - centro superior manuvrio")
        self.d = Punto(0, 0, 0, self.H,
                       "d - centro y centro de masa rueda delantera")
        # PARCIAL 2
        # centros de masa
        self.I = Subspace(x=0,
                          y=0,
                          z=0,
                          ax=0,
                          ay=0,
                          az=0,
                          parent_space=self.E,
                          nombre="I - centro de masa rueda trasera")
        self.J = Subspace(x=1.25 * self.r,
                          y=0,
                          z=2 * self.r,
                          ax=0,
                          ay=0,
                          az=0,
                          parent_space=self.D,
                          nombre="J - centro de masa marco bici")
        self.K = Subspace(x=0,
                          y=0,
                          z=-self.r,
                          ax=0,
                          ay=0,
                          az=0,
                          parent_space=self.G,
                          nombre="K - centro de masa tenedor")
        self.L = Subspace(x=0,
                          y=0,
                          z=0,
                          ax=0,
                          ay=0,
                          az=0,
                          parent_space=self.H,
                          nombre="L - centro de masa rueda delantera")
        # puntos nuevos
        self.e = Punto(0, 0, 0, self.J, "e - centro de masa marco bici")
        self.f = Punto(0, 0, 0, self.K, "f - centro de masa tenedor bici")
        # variables de masa
        self.mI = 1  # 1kg, alrededor de 2lb para una rueda de bicicleta
        self.mJ = 2  # 2kg, alrededor de 4.5lb para marco de bici
        self.mK = 1.4  # 1.4kg, Fox 32 fork weighs 3.06 pounds
        self.mL = self.mI  # otra rueda
        # matrices de inercia
        self.iI = MatrizDeInercia(
            [  # matriz de rueda trasera
                [0.5 * self.mI * self.r**2, 0, 0],  # (1/2)MR^2
                [0, self.mI * self.r**2, 0],  # MR^2
                [0, 0, 0.5 * self.mI * self.r**2],  # (1/2)MR^2
            ],
            self.I,
            self.mI,
            "iI")
        self.iJ = MatrizDeInercia(
            [  # matriz de marco bici
                [0, 0, 0],
                [0, (1 / 12) * self.mK * (self.r * 3)**2, 0],
                [0, 0, (1 / 12) * self.mK * (self.r * 3)**2],
            ],
            self.J,
            self.mJ,
            "iJ")
        self.iK = MatrizDeInercia(
            [  # matriz de tenedor
                [(1 / 12) * self.mK * (self.r * 3)**2, 0, 0],
                [0, (1 / 12) * self.mK * (self.r * 3)**2, 0],
                [0, 0, 0],
            ],
            self.K,
            self.mK,
            "iK")
        self.iL = MatrizDeInercia(
            [  # matriz de rueda delantera
                [0.5 * self.mL * self.r**2, 0, 0],  # (1/2)MR^2
                [0, self.mL * self.r**2, 0],  # MR^2
                [0, 0, 0.5 * self.mL * self.r**2],  # (1/2)MR^2
            ],
            self.L,
            self.mL,
            "iL")

    @property
    def m(self):
        """masa total / combinada"""
        return self.mI + self.mJ + self.mK + self.mL

    def mover(self):
        """mueve bici un paso"""
        for _ in range(0, 9):
            if (self.adjust_pitch() is False):
                break

        # encontrar velocidad linear de rueda trasera
        v_rueda_trasera = self.d_q4 * self.r
        # omega_rueda_delantera = 0

        if self.q6 == 0:
            # si angulo manuvrio == 0
            # centro instantaneo infinito
            # PARCIAL 1
            omega_rueda_delantera = self.d_q4
            v_rueda_delantera = v_rueda_trasera
            # alphas
            alpha_rueda_trasera = 0
            alpha_rueda_delantera = 0
            # se mueve en eje x de B unicamente
            self.update_pos_B(v_rueda_trasera * self.dt, 0, 0)
            # PARCIAL 2
            # v_marco_bici = v_rueda_trasera
        else:
            # si angulo manuvrio != 0
            # centro instantaneo existe
            # PARCIAL 1
            # encontrar centro instantaneo en B
            resp = self.encontrar_centro_instantaneo()
            # centro instantaneo
            ci_B = resp["centro_instantaneo_proyectado"]
            # crd_B = resp["centro_de_rueda_delantera_proyectado"]
            prd_B = resp["punto_de_contacto_rueda_delantera_proyectado"]
            crt_B = resp["centro_de_rueda_trasera_proyectado"]

            # cmb_B = resp["centro_de_marco_bici_proyectado"]
            # ct_B = resp["centro_de_tenedor_proyectado"]
            # distancia entre centro instantaneo y punto de
            # contacto rueda delantera
            r_delantera_piso = npl.norm(ci_B - prd_B)
            # distancia entre centro instantaneo y punto de
            # contacto rueda trasera
            r_trasera_piso = npl.norm(ci_B - crt_B)
            # print(f"{np.round(ci_B.llevar_a_parent_space(self.A),3)}")
            # velocidad angular centro instantaneo
            omega_ci = v_rueda_trasera / r_trasera_piso
            # calcular velocidad rueda delatera
            v_rueda_delantera = omega_ci * r_delantera_piso
            # calcular omega rueda delantera
            omega_rueda_delantera = v_rueda_delantera / self.r
            # vector omega centro instantaneo, rueda trasera y delantera
            v_omega_ci = omega_ci * Punto(0, 0, 1, self.B, "b3_B")
            v_omega_rt = Punto(0, 1, 0, self.E, "e2_E").llevar_a_parent_space(
                self.B) - self.E.origin.llevar_a_parent_space(self.B)
            v_omega_rd = Punto(0, 1, 0, self.H, "e2_H").llevar_a_parent_space(
                self.B) - self.H.origin.llevar_a_parent_space(self.B)
            # alpha rueda delantera
            alpha_rueda_delantera = npl.norm(v_omega_rd.cross(v_omega_ci))
            # alpha rueda trasera
            alpha_rueda_trasera = npl.norm(v_omega_rt.cross(v_omega_ci))
            # si volteando a izquierda
            if self.q6 > 0:
                # si esta volteando a la izquierda
                a_rotacion = omega_ci * self.dt
            else:
                # si esta volteando a la derecha
                a_rotacion = -omega_ci * self.dt

            ci_rotado_B = R.from_euler("z", a_rotacion,
                                       degrees=False).apply(ci_B)
            # calcular cambio en posicion
            dpos = ci_rotado_B - ci_B
            OB_A = self.B.origin.llevar_a_parent_space(self.A)
            dpos = Punto.from_list(
                dpos, self.B, "dpos_B").llevar_a_parent_space(self.A) - OB_A
            self.update_pos_B(dpos[0], dpos[1], a_rotacion)
            # PARCIAL 2
            # r_marco_bici = npl.norm(ci_B - cmb_B)
            # v_marco_bici = omega_ci * r_marco_bici
            # r_tenedor = npl.norm(ci_B - ct_B)
            # v_marco_bici = omega_ci * r_tenedor

        # agregar valores a lista

        # PARCIAL 1
        self.tiempo.append(self.update_time())
        # 1 - posicion x, y, z de punto de interes
        # 2 - velocidad de x, y, z de punto de interes
        punto_interes = self.E.origin.llevar_a_parent_space(
            self.A)  #, debug=True)
        self.update_punto_interes(punto_interes)
        # 3 - velocidad angular rueda trasera y delantera
        # 4 - aceleracion angular rueda trasera y delantera
        self.update_datos_ruedas(self.d_q4, omega_rueda_delantera,
                                 alpha_rueda_trasera, alpha_rueda_delantera)
        # print(f"w atras: {self.d_q4}\nw adelante: {omega_rueda_delantera}\n")
        # 5 - velocidad angular de interes
        # 6 - aceleracion angular de interes
        angulo_interes = omega_ci if omega_ci is not None else 0
        self.update_angulos_interes(angulo_interes, -1 if self.q6 < 0 else 1)

        # PARCIAL 2
        iI = self.iI.llevar_a_origen_de_parent_space(self.A)
        iJ = self.iJ.llevar_a_origen_de_parent_space(self.A)
        iK = self.iK.llevar_a_origen_de_parent_space(self.A)
        iL = self.iL.llevar_a_origen_de_parent_space(self.A)
        # 1 - centro de masa general respecto a marco fijo
        centro_masa = self.encontrar_centro_de_masa_en_B(
        ).llevar_a_parent_space(self.A)
        self.update_centro_masa(centro_masa)
        # 2 inercias lineares y angulares
        self.update_escalares_inercia(iI, iJ, iK, iL)
        #
        p_CM_en_base = self.encontrar_centro_de_masa_en_B(
        ).llevar_a_parent_space(self.base)
        p_CM = self.encontrar_centro_de_masa_en_B().llevar_a_parent_space(
            self.A)
        p_I = self.I.origin.llevar_a_parent_space(self.A)
        p_J = self.J.origin.llevar_a_parent_space(self.A)
        p_K = self.K.origin.llevar_a_parent_space(self.A)
        p_L = self.L.origin.llevar_a_parent_space(self.A)
        self.update_datos_CM(p_CM, p_I, p_J, p_K, p_L, p_CM_en_base)

    def encontrar_centro_de_masa_en_B(self) -> "Punto":
        """"encontrar centro de masa"""
        # obtener vectores que apuntan a cada centro en B
        OI_B = self.I.origin.llevar_a_parent_space(self.B)
        OJ_B = self.J.origin.llevar_a_parent_space(self.B)
        OK_B = self.K.origin.llevar_a_parent_space(self.B)
        OL_B = self.L.origin.llevar_a_parent_space(self.B)
        i = OI_B
        j = OJ_B
        k = OK_B
        l = OL_B
        # basado en la masa de cada uno, encontrar vector de centro de masa general
        # masa total
        M = self.mI + self.mJ + self.mK + self.mL
        # calcular cordenadas individuales
        x_cm = (self.mI * i.x + self.mJ * j.x + self.mK * k.x +
                self.mL * l.x) / M
        y_cm = (self.mI * i.y + self.mJ * j.y + self.mK * k.y +
                self.mL * l.y) / M
        z_cm = (self.mI * i.z + self.mJ * j.z + self.mK * k.z +
                self.mL * l.z) / M
        return Punto(x_cm, y_cm, z_cm, self.B, "centro_masa_B")

    class __encontrar_centro_instantaneo__(TypedDict):
        """tipo de retorno de encontrar_centro_instantaneo()"""
        # PARCIAL 1
        centro_de_rueda_trasera_proyectado: Punto
        centro_de_rueda_delantera_proyectado: Punto
        punto_de_contacto_rueda_delantera_proyectado: Punto
        centro_instantaneo_proyectado: Punto
        # PARCIAL 2
        centro_de_marco_bici_proyectado: Punto
        centro_de_tenedor_proyectado: Punto

    def encontrar_centro_instantaneo(self) -> __encontrar_centro_instantaneo__:
        """retorna centro instantaneo en B"""
        # si volteando a izquierda apuntar a izquierda
        # si volteando a derecha apuntar a derecha
        y = -1 if self.q6 > 0 else 1
        # d2 en D es perpendicular a rueda trasera
        # tambien esta en plano xy de D en D
        v1_D = Punto(0, y, 0, self.D, "d2_D")
        # h2 en H es perpendicular a rueda delantera
        h2_H = Punto(0, y, 0, self.H, "h2_H")
        OH_H = self.H.origin

        # llevar h2_H a D
        OH_D = OH_H.llevar_a_parent_space(self.D)
        h2_D = h2_H.llevar_a_parent_space(self.D) - OH_D
        v2_D = h2_D.project_onto_plane(Punto(0, 0, 1, self.D, "d3_D"),
                                       parent_space=self.D)
        # encontrar angulo entre b1_B y v2_B
        a2 = v1_D.angulo_entre(v2_D)
        # encontrar angulo entre vectores
        # a2 = v1_B.angulo_entre(v2_B)
        # distancia entre puntos b y d
        d = npl.norm(
            self.b.llevar_a_parent_space(self.D) -
            self.d.llevar_a_parent_space(self.D))
        # encontrar longitud de v1_B y v2_B
        norm_v1 = np.abs(d / np.tan(a2))
        norm_v2 = np.abs(d / np.sin(a2))
        # aplicar
        v1_D: Punto = v1_D * norm_v1
        v2_D: Punto = v2_D * norm_v2
        # notas hasta ahora:
        # - v1_D va de origen de marco D a centro instantaneo
        #   en plano xy de D en D
        # - v2_D va de centro de rueda delantera hasta centro
        #   instantaneo en plano xy de D en D
        # v3_D es vector de centro de rueda trasera a centro de rueda
        # delantera proyectado en plano xy de D en D
        v3_D = self.d.llevar_a_parent_space(
            self.D) - self.b.llevar_a_parent_space(self.D)
        v3_D = v3_D.project_onto_plane(n=Punto(0, 0, 1, self.D, "d3_D"),
                                       parent_space=self.D)
        # llevar puntos a marco B
        v1_B = v1_D.llevar_a_parent_space(self.B)
        v2_B = v2_D.llevar_a_parent_space(self.B)
        v3_B = v3_D.llevar_a_parent_space(self.B)
        # proyectar puntos a plano xy de B
        b3_B = Punto(0, 0, 1, self.B, "b3_B")
        # v1_B va de centro de rueda trasera hasta centro
        # instantaneo proyectado en plano xy de B en B
        v1_B = v1_B.project_onto_plane(n=b3_B, parent_space=self.B)
        # v2_B va de centro de rueda delantera hasta centro
        # instantaneo proyectado en plano xy de B en B
        v2_B = v2_B.project_onto_plane(n=b3_B, parent_space=self.B)
        # v3_B va de centro de rueda trasera a centro de rueda
        # delantera proyectado en el plano xy de B en B
        v3_B = v3_B.project_onto_plane(n=b3_B, parent_space=self.B)
        # v4_B va de origen de B a centro de rueda trasera
        # proyectado en plano xy de B en B
        v4_B = self.b.llevar_a_parent_space(self.B).project_onto_plane(
            n=b3_B, parent_space=self.B)
        # organizar vectores
        # origen B a centro de rueda delantera
        v5_B = v4_B + v3_B
        # origen B a centro instantaneo
        v6_B = v4_B + v1_B
        # encontrar angulo entre b3_B y plano xz de H, para esto:
        # h3 en B
        h2_B = Punto(0, 1, 0, self.H, "h2_H").llevar_a_parent_space(self.B)
        # proyectar b3_B a plano xz de H en B
        v7_B = b3_B.project_onto_plane(n=h2_B, parent_space=self.B)
        # angulo entre b3_B y v7_B
        a3 = b3_B.angulo_entre(v7_B)
        # distancia entre centro de rueda proyectado en B
        # y punto de contacto en piso
        d = self.r * np.sin(a3)
        # v2_B unitario
        v8_B = (v2_B / npl.norm(v2_B)) * -1
        # para que vaya de centro instantaneo a punto de contacto
        # de ruedo delantera con piso
        v8_B = v8_B = v8_B * (npl.norm(v2_B) + d)
        v9_B = v8_B + v6_B
        # debug
        # print(f"ci_B: {np.round(v6_B,2)}, v9_B: {np.round(v5_B,2)}")

        # PARCIAL 2
        # centro de marco de bici proyectado a plano xy de B
        v10_B = self.J.origin.llevar_a_parent_space(self.B)
        v10_B = v10_B.project_onto_plane(n=Punto(0, 0, 1, self.B, "b3_B"),
                                         parent_space=self.B)
        # centro de tenedor proyectado a plano xy de B
        v11_B = self.K.origin.llevar_a_parent_space(self.B)
        v11_B = v11_B.project_onto_plane(n=Punto(0, 0, 1, self.B, "b3_B"),
                                         parent_space=self.B)

        # retornar
        return {
            "centro_de_rueda_trasera_proyectado": v4_B,
            "centro_de_rueda_delantera_proyectado": v5_B,
            "punto_de_contacto_rueda_delantera_proyectado": v9_B,
            "centro_instantaneo_proyectado": v6_B,
            "centro_de_marco_bici_proyectado": v10_B,
            "centro_de_tenedor_proyectado": v11_B
        }

    def adjust_pitch(self) -> bool:
        """
        adjusts pitch so front wheel touches ground
        needs to be called at least 4 times
        """
        # vector normal a plano xz de H en H
        h2_H = Punto(0, 1, 0, self.H, "h2_H")
        # origen de H en H
        OH_H = Punto(0, 0, 0, self.H, "Oh_H")
        # origen de H en B
        OH_B = OH_H.llevar_a_parent_space(self.B)
        # vector normal a plano xz de H en B
        h2_B = h2_H.llevar_a_parent_space(self.B) - OH_B
        # vector hacia abajo en B
        v1_B = Punto(0, 0, -1, self.C, "v1_B").llevar_a_parent_space(self.B)
        # proyectar v1_B a plano xz de H en B
        v2_B = v1_B.project_onto_plane(n=h2_B, parent_space=self.B)
        # llevar a punto mas inferior posible en rueda delantera
        v2_B: Punto = (v2_B / npl.norm(v2_B)) * self.r
        # ubicar con respecto a marco B
        v3_B: Punto = v2_B + self.d.llevar_a_parent_space(self.B)
        # vector normal a plano xz de D en D
        d2_D = Punto(0, 1, 0, self.D, "d2_D")
        # origen D en D
        OD_D = Punto(0, 0, 0, self.D, "OD_D")
        # origen D en B
        OD_B = OD_D.llevar_a_parent_space(self.B)
        # vector normal a plano xz de D en B
        d2_B: Punto = d2_D.llevar_a_parent_space(self.B) - OD_B
        # llevar v3_D a plano xz de D en B
        v3_B = v3_B.project_onto_plane(n=d2_B, parent_space=self.B)
        # vector unitario de B en B
        b1_B = Punto(1, 0, 0, self.B, "b1_B")
        # llevar b1_B a plano xz de D en B
        b1_B = b1_B.project_onto_plane(n=d2_B, parent_space=self.B)
        # angulo entre v1_D y b1_B es ajuste de pitch
        ajuste = v3_B.angulo_entre(b1_B)
        if (ajuste <= 0.0000001): return False
        if v3_B.z > 0:
            ajuste = -ajuste
        self.set_q3(self.q3 + ajuste)
        return True

    def set_q3(self, rads: float):
        """actualizar q3 - pitch bici"""
        self.q3 = rads
        self.D.ay = rads

    def update_pos_B(self, dx, dy, dq1):
        """actualizar posicion marco B"""
        self.B.x += dx
        self.B.y += dy
        self.q1 += dq1
        self.q1 = np.fmod(self.q1, 2 * pi)
        # print(f"dq1: {dq1}")
        self.B.az = self.q1

    def update_time(self) -> float:
        """actualizar tiempo"""
        t = self.t
        self.t += self.dt
        return t

    # 1 - posicion x, y, z de punto de interes
    # 2 - velocidad de x, y, z de punto de interes
    def update_punto_interes(self, punto_interes: Punto):
        """guarda posicion de punto de interes"""
        # cordenadas
        self.punto_de_interes_x.append(punto_interes.x)
        self.punto_de_interes_y.append(punto_interes.y)
        self.punto_de_interes_z.append(punto_interes.z)
        # velocidad de cordenadas
        self.punto_de_interes_dx = np.diff(self.punto_de_interes_x)
        self.punto_de_interes_dy = np.diff(self.punto_de_interes_y)
        self.punto_de_interes_dz = np.diff(self.punto_de_interes_z)
        # aceleracion de cordenadas
        self.punto_de_interes_ddx = np.diff(self.punto_de_interes_dx)
        self.punto_de_interes_ddy = np.diff(self.punto_de_interes_dy)
        self.punto_de_interes_ddz = np.diff(self.punto_de_interes_dz)

    @staticmethod
    def __diff_angulos__(l: list, debug=False):
        diff = []
        for i in range(0, len(l) - 1):
            a2 = l[i + 1]
            a1 = l[i]
            d = np.fmod(np.fmod(a2 - a1, 2 * pi) + 2 * pi, 2 * pi)
            d = np.min([d, 2 * pi - d])
            diff.append(d)
            if debug: print(f"a1: {a1}, a2: {a2}, d: {d}")
        return diff

    # 3 - velocidad angular rueda trasera y delantera
    # 4 - aceleracion angular rueda trasera y delantera
    def update_datos_ruedas(self, omega_trasera, omega_delantera,
                            alpha_rueda_trasera, alpha_rueda_delantera):
        """actualiza datos de rueda trasera y delantera"""
        # velocidades angulares
        self.omega_rueda_trasera.append(omega_trasera)
        self.omega_rueda_delantera.append(omega_delantera)
        # aceleraciones
        self.alpha_rueda_trasera.append(alpha_rueda_trasera)
        self.alpha_rueda_delantera.append(alpha_rueda_delantera)

    # 5 - velocidad angular de interes
    # 6 - aceleracion angular de interes
    def update_angulos_interes(self, angulo_interes: float, signo: -1 | 1):
        """actualiza angulo de interes, y su primera y segunda derivada"""
        # print(f"angulo interes: {angulo_interes}")
        self.omega_de_interes.append(angulo_interes * signo)
        self.alpha_de_interes = np.divide(
            Bicycle.__diff_angulos__(self.omega_de_interes), self.dt)

    # 1 - centro de masa general
    def update_centro_masa(self, centro_masa: Punto):
        """actualiza centro de masa"""
        self.centro_masa_x.append(centro_masa.x)
        self.centro_masa_y.append(centro_masa.y)
        self.centro_masa_z.append(centro_masa.z)

    def update_escalares_inercia(self, iI: MatrizDeInercia,
                                 iJ: MatrizDeInercia, iK: MatrizDeInercia,
                                 iL: MatrizDeInercia):
        """actualizar escalares de inercia"""
        # actualizar matrices
        self.matriz_inercia_I.append(iI)
        self.matriz_inercia_J.append(iJ)
        self.matriz_inercia_K.append(iK)
        self.matriz_inercia_L.append(iL)

    class __datos_vista_aerea__(TypedDict):
        centro_masa: list[float]
        centro_rd: list[float]
        centro_rt: list[float]
        centro_instantaneo: list[list[float]]
        rueda_delantera: list[list[float]]
        rueda_trasera: list[list[float]]

    def datos_vista_aerea(self) -> __datos_vista_aerea__:
        """retorna puntos/lineas en vista aerea"""
        centro_masa = self.encontrar_centro_de_masa_en_B(
        ).llevar_a_parent_space(self.A)
        centro_rd = self.d.llevar_a_parent_space(self.A)
        centro_rt = self.b.llevar_a_parent_space(self.A)
        if ():
            centro_instantaneo = self.encontrar_centro_instantaneo(
            )["centro_instantaneo_proyectado"].llevar_a_parent_space(self.A)
        else:
            centro_instantaneo = self.centro_instantaneo_Parcial3(
            )["ci"].llevar_a_parent_space(self.A)
        # vector que apunta en direccion de rueda delantera
        vrd = Punto(1, 0, 0, self.G, "u_x G").llevar_a_parent_space(
            self.A) - self.G.origin.llevar_a_parent_space(self.A)
        # print(f"vrd: {np.round(vrd,3)}")
        vrd.project_onto_plane(
            Punto(1, 0, 0, self.A, "").cross(Punto(0, 1, 0, self.A, "")),
            self.A, "")
        vrd = vrd / npl.norm(vrd)
        # punto delantero rueda delantera
        pdrd = centro_rd + vrd * self.r
        ptrd = centro_rd + vrd * -self.r
        # vector que apunta en direccion de rueda trasera
        vrt = Punto(1, 0, 0, self.B, "").llevar_a_parent_space(
            self.A) - self.B.origin.llevar_a_parent_space(self.A)
        vrt.project_onto_plane(
            Punto(1, 0, 0, self.A, "").cross(Punto(0, 1, 0, self.A, "")),
            self.A, "")
        vrt = vrt / npl.norm(vrt)
        # punto delantero rueda delantera
        pdrt = centro_rt + vrt * self.r
        ptrt = centro_rt + vrt * -self.r
        # en nuestro caso y se invierte, de vista aerea esta definido
        # en la direccion opuesta
        return {
            "centro_masa": [centro_masa.x, -centro_masa.y],
            "centro_rd": [centro_rd.x, -centro_rd.y],
            "centro_rt": [centro_rt.x, -centro_rt.y],
            "centro_instantaneo":
            [centro_instantaneo[0], -centro_instantaneo[1]],
            "rueda_delantera": [[ptrd[0], -ptrd[1]], [pdrd[0], -pdrd[1]]],
            "rueda_trasera": [[ptrt[0], -ptrt[1]], [pdrt[0], -pdrt[1]]],
        }

    def plot_vista_aerea(self, fig: go.Figure, row=-1, col=-1, rx=6, ry=5):
        """grafica puntos de bici en vista aerea"""
        d = self.datos_vista_aerea()
        cm = d["centro_masa"]
        crd = d["centro_rd"]
        crt = d["centro_rt"]
        ci = d["centro_instantaneo"]
        rd = d["rueda_delantera"]
        rt = d["rueda_trasera"]
        # centro masa
        fig.append_trace(go.Scatter(name="centro masa",
                                    x=[cm[0]],
                                    y=[cm[1]],
                                    marker_color="rgba(255, 0, 0, 1)"),
                         row=row,
                         col=col)
        # centro rueda delantera
        fig.append_trace(go.Scatter(name="centro rueda delantera",
                                    x=[crd[0]],
                                    y=[crd[1]],
                                    marker_color="rgba(17, 133, 17, 1)"),
                         row=row,
                         col=col)
        # centro rueda trasera
        fig.append_trace(go.Scatter(name="centro rueda trasera",
                                    x=[crt[0]],
                                    y=[crt[1]],
                                    marker_color="rgba(0, 0, 255, 1)"),
                         row=row,
                         col=col)
        # centro instantaneo
        fig.append_trace(go.Scatter(name="centro instantaneo",
                                    x=[ci[0]],
                                    y=[ci[1]],
                                    marker_color="rgba(0, 0, 255, 1)"),
                         row=row,
                         col=col)
        # rueda delantera
        fig.append_trace(go.Scatter(name="rueda delantera",
                                    x=[p[0] for p in rd],
                                    y=[p[1] for p in rd],
                                    marker_color="rgba(17, 133, 17, 0.5)"),
                         row=row,
                         col=col)
        # rueda trasera
        fig.append_trace(go.Scatter(name="rueda trasera",
                                    x=[p[0] for p in rt],
                                    y=[p[1] for p in rt],
                                    marker_color="rgba(0, 0, 255, 0.5)"),
                         row=row,
                         col=col)
        # configure axes
        fig.update_xaxes(range=[-rx, rx],
                         row=row,
                         col=col,
                         overwrite=True,
                         minallowed=-rx,
                         maxallowed=rx,
                         fixedrange=True)
        fig.update_yaxes(range=[-ry, ry],
                         row=row,
                         col=col,
                         overwrite=True,
                         minallowed=-ry,
                         maxallowed=ry,
                         fixedrange=True)

    def update_datos_CM(self, p_CM, p_I, p_J, p_K, p_L, p_CM_en_base):
        """actualiza datos de centros de masa"""
        self.pos_CM.append(p_CM)
        self.pos_I.append(p_I)
        self.pos_J.append(p_J)
        self.pos_K.append(p_K)
        self.pos_L.append(p_L)
        self.pos_CM_base.append(p_CM_en_base)
        # vel
        self.vel_CM = [
            list(i) for i in (np.diff([p[0] for p in self.pos_CM]),
                              np.diff([p[1] for p in self.pos_CM]),
                              np.diff([p[2] for p in self.pos_CM]))
        ]
        self.vel_I = [
            list(i) for i in zip(np.diff([p[0] for p in self.pos_I]),
                                 np.diff([p[1] for p in self.pos_I]),
                                 np.diff([p[2] for p in self.pos_I]))
        ]
        self.vel_J = [
            list(i) for i in zip(np.diff([p[0] for p in self.pos_J]),
                                 np.diff([p[1] for p in self.pos_J]),
                                 np.diff([p[2] for p in self.pos_J]))
        ]
        self.vel_K = [
            list(i) for i in zip(np.diff([p[0] for p in self.pos_K]),
                                 np.diff([p[1] for p in self.pos_K]),
                                 np.diff([p[2] for p in self.pos_K]))
        ]
        self.vel_L = [
            list(i) for i in zip(np.diff([p[0] for p in self.pos_L]),
                                 np.diff([p[1] for p in self.pos_L]),
                                 np.diff([p[2] for p in self.pos_L]))
        ]
        self.vel_CM_base = [
            list(i) for i in zip(np.diff([p[0] for p in self.pos_CM_base]),
                                 np.diff([p[1] for p in self.pos_CM_base]),
                                 np.diff([p[2] for p in self.pos_CM_base]))
        ]
        # acc
        self.acc_CM = [
            list(i) for i in zip(np.diff([p[0] for p in self.vel_CM]),
                                 np.diff([p[1] for p in self.vel_CM]),
                                 np.diff([p[2] for p in self.vel_CM]))
        ]
        self.acc_I = [
            list(i) for i in zip(np.diff([p[0] for p in self.vel_I]),
                                 np.diff([p[1] for p in self.vel_I]),
                                 np.diff([p[2] for p in self.vel_I]))
        ]
        self.acc_J = [
            list(i) for i in zip(np.diff([p[0] for p in self.vel_J]),
                                 np.diff([p[1] for p in self.vel_J]),
                                 np.diff([p[2] for p in self.vel_J]))
        ]
        self.acc_K = [
            list(i) for i in zip(np.diff([p[0] for p in self.vel_K]),
                                 np.diff([p[1] for p in self.vel_K]),
                                 np.diff([p[2] for p in self.vel_K]))
        ]
        self.acc_L = [
            list(i) for i in zip(np.diff([p[0] for p in self.vel_L]),
                                 np.diff([p[1] for p in self.vel_L]),
                                 np.diff([p[2] for p in self.vel_L]))
        ]
        self.acc_CM_base = [
            list(i) for i in zip(np.diff([p[0] for p in self.vel_CM_base]),
                                 np.diff([p[1] for p in self.vel_CM_base]),
                                 np.diff([p[2] for p in self.vel_CM_base]))
        ]

    def nivelarParcial3(self):
        """nivela bicicleta (cambia z de subespacio B)"""
        # centro rueda trasera en A
        crt = self.b.llevar_a_parent_space(self.base)
        h = crt.z
        ajuste = self.r - h
        self.B.z += ajuste
        self.B.x -= crt.x
        # crt ahora en base
        crt = self.b.llevar_a_parent_space(self.base)
        # ahora encontrar y retornar punto de contacto con el piso en B
        # unitario en base hacia abajo
        abajo = Punto(0, 0, -1, self.base, "abajo unitario en B")
        # piso rueda trasera en base
        prt = crt + abajo * self.r
        # print(f"prt en base: {np.round(prt,3)}")
        # asegurar que este en espacio correcto y retornar
        return Punto.from_list(prt, self.base, "prt_B")

    class __centro_instantaneo_Parcial3__(TypedDict):
        """tipo de retorno"""
        # vector de radio centro instantaneo desde rueda trasera a centro instantaneo
        vrci: Punto
        # ubicacion de centro instantaneo con respecto a origen base en base
        ci: Punto
        # cambio angulo
        d_angulo: float

    def centro_instantaneo_Parcial3(self) -> __centro_instantaneo_Parcial3__:
        """encuentra puntos relacionados a centro instantaneo"""
        prt = self.nivelarParcial3()
        # CENTRO INSTANTANEO
        # encontrar velocidad rueda trasera (v = omega * r)
        vrt = self.d_q4 * self.r
        # encontrar diametro de centro instantaneo dado longitud de arco y cambio en angulo
        # en 1 segundo
        d_angulo = np.abs(self.d_q1 * self.dt)
        l_arco = vrt * self.dt
        # circumference
        C = (l_arco * 2 * pi) / d_angulo
        # radio centro instantaneo
        rci = C / (2 * pi)
        # direccion centro instantaneo desde origen base en base
        dci = Punto(0, 1 if self.d_q1 < 0 else -1, 0, self.base, "dci")
        # vector de radio centro instantaneo
        vrci = dci * rci
        # print(f"dci: {rci}, d_angulo: {d_angulo}, vrt: {vrt}")
        # ubicacion de centro instantaneo con respecto a base
        ajuste_ci = Punto(prt.x, prt.y, 0, self.base, "ajuste ci")
        ci = vrci + ajuste_ci
        return {
            "vrci": vrci,
            "ci": ci,
            "d_angulo": d_angulo,
        }

    def mover_parcial_3(self):
        """mueve bicicleta si esta parada en una rueda"""
        # primero que todo, encontrar punto de contacto con piso en base, y nivelar bici
        # solo es cuestion de cambiar z de subespacio B
        prt = self.nivelarParcial3()
        # MOMENTOS DE INERCIA / MATRIZ COMBINADA
        # luego, llevar todas las matrices de inercia al punto de contacto
        I_base = self.iI.llevar_a_origen_de_parent_space(self.base)
        J_base = self.iI.llevar_a_origen_de_parent_space(self.base)
        K_base = self.iI.llevar_a_origen_de_parent_space(self.base)
        L_base = self.iI.llevar_a_origen_de_parent_space(self.base)
        # combinar matrices de inercia, tambien suma masa
        # se pueden combinar por lo que estan ubicados en la misma base y en el mismo punto
        combinado = I_base + J_base + K_base + L_base
        # FUERZAS
        # ahora encontrar fuerza en centro de masa
        g = 9.81  # m/s^2
        F_peso = self.m * g
        # ahora fuerza debida a aceleracion
        if len(self.acc_CM) > 0:
            # fuerza es aceleracion centro de masa *
            F_acc = self.m * npl.norm(self.acc_CM_base[0])
        else:
            # si no hay datos para fuerza todavia entonces 0
            F_acc = 0
        # vector de peso fuerza
        # vector hacia abajo en base
        abajo = Punto(0, 0, -1, self.base, "abajo en base")
        vF_peso = abajo * F_peso
        # vector de fuerza acc
        # vector de acc
        if len(self.acc_CM) > 0:
            acc = self.acc_CM_base[-1]
        else:
            acc = [0, 0, 0]
        ## TODO: vectores de aceleracion estan con especto a piso (A)
        ## toca hacer una nueva lista de pos, vel, y acc centro de masa pero
        # con respecto a base
        acc = Punto.from_list(
            acc,
            self.base,
            "acc en base",
        ).unitario()
        # vector fuerza debida a aceleracion
        vF_acc = acc * F_acc
        # vector fuerza total
        vF_total = vF_peso + vF_acc
        # ya encontrada fuerza total, encontrar vector entre centro masa y punto piso
        cm = self.encontrar_centro_de_masa_en_B().llevar_a_parent_space(
            self.base)
        # prt = prt
        # radio fuerza
        rF = cm - prt
        print(f"acc_CM_base: {self.acc_CM_base}")
        # print(f"acc: {acc}, F_acc: {vF_acc}")
        # encontrar torque total
        T = rF.cross(vF_total, parent_space=self.base)
        # vector de torque obtenido, separar en componentes
        self.actualizar_torque_parcial3(T)

        # CENTRO INSTANTANEO
        datos = self.centro_instantaneo_Parcial3()
        # vector de radio centro instantaneo
        vrci = datos["vrci"]
        # ubicacion de centro instantaneo con respecto a base
        ci = datos["ci"]
        # d_angulo
        d_q1 = datos["d_angulo"]
        # print(f"ci: {np.round(ci, 3)}")
        # print(f"vrci: {vrci}, dq1: {d_q1}")
        # rotar centro instantaneo
        rot = R.from_euler("z",
                           d_q1 if self.d_q1 > 0 else -d_q1,
                           degrees=False)
        nueva_posicion = Punto.from_list(rot.apply(-vrci), self.base,
                                         "nueva pos en base")
        # cambio en posicion
        O = self.base.origin.llevar_a_parent_space(self.A)
        dpos = (nueva_posicion - (-vrci)).llevar_a_parent_space(self.A) - O
        # actualizar pos
        # print(f"dpos: {np.round(dpos,3)}")
        # print(f"dpos: {dpos}")
        self.actualizar_pos_parcial3(dpos, d_q1)
        # actualizar valores en listas parcial 3
        self.valores_parcial_3()

    def actualizar_pos_parcial3(self, dpos: Punto, d_q1: float):
        """actualizar pos bici"""
        # print(f"dpos: {dpos}")
        self.base.x += dpos.x
        self.base.y += dpos.y
        # si mas grande que 2 pi entonces reducir
        if self.base.az > (2 * pi) or self.base.az < (-2 * pi):
            self.base.az = np.fmod(self.base.az, 2 * pi)
        self.base.az += d_q1

    def actualizar_torque_parcial3(self, T: Punto):
        """actualiza listas de torque"""
        # print(f"torque antes: {T}")
        O = T.parent_space.origin.llevar_a_parent_space(self.A)
        T = T.llevar_a_parent_space(self.A) - O
        # print(f"torque: {T}")
        self.torque_total.append(T)

    def valores_parcial_3(self):
        """mueve bici un paso"""
        # encontrar velocidad linear de rueda trasera
        v_rueda_trasera = self.d_q4 * self.r
        # omega_rueda_delantera = 0

        if self.q6 == 0:
            # si angulo manuvrio == 0
            # centro instantaneo infinito
            # PARCIAL 1
            omega_rueda_delantera = self.d_q4
            v_rueda_delantera = v_rueda_trasera
            # alphas
            alpha_rueda_trasera = 0
            alpha_rueda_delantera = 0
            omega_ci = 0
            # se mueve en eje x de B unicamente
            # PARCIAL 2
            # v_marco_bici = v_rueda_trasera
        else:
            # si angulo manuvrio != 0
            # centro instantaneo existe
            # PARCIAL 1
            # encontrar centro instantaneo en B
            resp = self.encontrar_centro_instantaneo()
            # centro instantaneo
            ci_B = resp["centro_instantaneo_proyectado"]
            # crd_B = resp["centro_de_rueda_delantera_proyectado"]
            prd_B = resp["punto_de_contacto_rueda_delantera_proyectado"]
            crt_B = resp["centro_de_rueda_trasera_proyectado"]

            # cmb_B = resp["centro_de_marco_bici_proyectado"]
            # ct_B = resp["centro_de_tenedor_proyectado"]
            # distancia entre centro instantaneo y punto de
            # contacto rueda delantera
            r_delantera_piso = npl.norm(ci_B - prd_B)
            # distancia entre centro instantaneo y punto de
            # contacto rueda trasera
            r_trasera_piso = npl.norm(ci_B - crt_B)
            # print(f"{np.round(ci_B.llevar_a_parent_space(self.A),3)}")
            # velocidad angular centro instantaneo
            omega_ci = v_rueda_trasera / r_trasera_piso
            # calcular velocidad rueda delatera
            v_rueda_delantera = omega_ci * r_delantera_piso
            # calcular omega rueda delantera
            omega_rueda_delantera = v_rueda_delantera / self.r
            # vector omega centro instantaneo, rueda trasera y delantera
            v_omega_ci = omega_ci * Punto(0, 0, 1, self.B, "b3_B")
            v_omega_rt = Punto(0, 1, 0, self.E, "e2_E").llevar_a_parent_space(
                self.B) - self.E.origin.llevar_a_parent_space(self.B)
            v_omega_rd = Punto(0, 1, 0, self.H, "e2_H").llevar_a_parent_space(
                self.B) - self.H.origin.llevar_a_parent_space(self.B)
            # alpha rueda delantera
            alpha_rueda_delantera = npl.norm(v_omega_rd.cross(v_omega_ci))
            # alpha rueda trasera
            alpha_rueda_trasera = npl.norm(v_omega_rt.cross(v_omega_ci))
            # si volteando a izquierda
            if self.q6 > 0:
                # si esta volteando a la izquierda
                a_rotacion = omega_ci * self.dt
            else:
                # si esta volteando a la derecha
                a_rotacion = -omega_ci * self.dt
            # calcular cambio en posicion

        # agregar valores a listas

        # PARCIAL 1
        self.tiempo.append(self.update_time())
        # 1 - posicion x, y, z de punto de interes
        # 2 - velocidad de x, y, z de punto de interes
        punto_interes = self.E.origin.llevar_a_parent_space(
            self.A)  #, debug=True)
        self.update_punto_interes(punto_interes)
        # 3 - velocidad angular rueda trasera y delantera
        # 4 - aceleracion angular rueda trasera y delantera
        self.update_datos_ruedas(self.d_q4, omega_rueda_delantera,
                                 alpha_rueda_trasera, alpha_rueda_delantera)
        # print(f"w atras: {self.d_q4}\nw adelante: {omega_rueda_delantera}\n")
        # 5 - velocidad angular de interes
        # 6 - aceleracion angular de interes
        angulo_interes = omega_ci if omega_ci is not None else 0
        self.update_angulos_interes(angulo_interes, -1 if self.q6 < 0 else 1)

        # PARCIAL 2
        iI = self.iI.llevar_a_origen_de_parent_space(self.A)
        iJ = self.iJ.llevar_a_origen_de_parent_space(self.A)
        iK = self.iK.llevar_a_origen_de_parent_space(self.A)
        iL = self.iL.llevar_a_origen_de_parent_space(self.A)
        # 1 - centro de masa general respecto a marco fijo
        centro_masa = self.encontrar_centro_de_masa_en_B(
        ).llevar_a_parent_space(self.A)
        self.update_centro_masa(centro_masa)
        # 2 inercias lineares y angulares
        self.update_escalares_inercia(iI, iJ, iK, iL)
        #
        p_CM_en_base = self.encontrar_centro_de_masa_en_B(
        ).llevar_a_parent_space(self.A)
        p_CM = self.encontrar_centro_de_masa_en_B().llevar_a_parent_space(
            self.A)
        p_I = self.I.origin.llevar_a_parent_space(self.A)
        p_J = self.J.origin.llevar_a_parent_space(self.A)
        p_K = self.K.origin.llevar_a_parent_space(self.A)
        p_L = self.L.origin.llevar_a_parent_space(self.A)
        self.update_datos_CM(p_CM, p_I, p_J, p_K, p_L, p_CM_en_base)


# parcial 1 - velocidades
# parcial 2 - inercia
# parcial 3 - cinetica
modo = "parcial 3"
# crear bici
bici_dict = {
    "bici":
    Bicycle(
        q1=0,  # bike yaw
        q2=0,  #pi / 4,  # bike roll
        q3=pi / 4,  #pi / 8,  # bike pitch
        q4=0,  # back wheel angle
        q5=0,  #pi / 8,  # fork pitch angle en rango [0, -pi/4]
        q6=0,  # fork yaw
        q7=0,  # front wheel angle
        d_q1=pi / 4,
        d_q4=pi / 4,  # back wheel angular speed
        dt=0.5,  # time interval
        r=0.35,  # radius bike wheel, for 27.5" bike wheel
    )
}
print("bici creada")

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='grafica',
        #   responsive=True,
        style={
            'autosize': False,
            'height': 825,
            'width': 1300,
            'margin': {
                "t": "0",
                "b": "0",
                "l": "0",
                "r": "0"
            }
        },
        mathjax=True),
    dcc.Interval(
        id="actualizar-bici",
        interval=1000,  # ms
        n_intervals=0,
    )
])


@callback(Output("grafica", "figure"), Input("actualizar-bici", "n_intervals"))
def update_graph(_):
    """update graph"""
    bici = bici_dict["bici"]
    if modo == "parcial 1":
        bici.mover()
        return graficar_parcial_1()
    if modo == "parcial 2":
        bici.mover()
        return graficar_parcial_2()
    if modo == "parcial 3":
        bici.mover_parcial_3()
        return graficar_parcial_3()
    return None


def graficar_parcial_3():
    """graficar parcial 1"""
    # graph setup
    bici = bici_dict["bici"]
    fig = make_subplots(
        rows=3,
        cols=4,
        shared_xaxes=False,
        subplot_titles=[
            r"$x\text{ todos cm}$",
            r"$y\text{ todos cm}$",
            r"$z\text{ todos cm}$",
            r"$\text{vista superior}$",
            ##############################
            r"$x'\text{ todos cm}$",
            r"$y'\text{ todos cm}$",
            r"$z'\text{ todos cm}$",
            r"$\text{torque total}$",
            ##############################
            r"$x''\text{ todos cm}$",
            r"$y''\text{ todos cm}$",
            r"$z''\text{ todos cm}$",
            r"$$",
        ])
    fig.update_yaxes(selector=dict(type="scatter"),
                     autorangeoptions=dict(include=[0, 1]))
    # sub figures
    # vista aerea bici
    bici.plot_vista_aerea(fig, row=1, col=4, rx=5, ry=4)
    # tiempo
    t = list(bici.tiempo)[:bici.n_datos - 2]
    # posicion centros de masa
    for c in range(1, 4):
        # for each column
        datos = zip(
            [bici.pos_CM, bici.pos_I, bici.pos_J, bici.pos_K, bici.pos_L],
            [r"$pos_{CM}$", r"$pos_I$", r"$pos_J$", r"$pos_K$", r"$pos_L$"],
            [
                "rgba(255,105,180,1)",
                "rgba(255, 0, 0, 1)",
                "rgba(17, 133, 17, 1)",
                "rgba(0, 0, 255, 1)",
                "rgba(0, 0, 0, 1)",
            ],
        )
        # plot 1 coordinate for each data set
        for torque, nombre, color in datos:
            # print(f"c: {c}")
            fig.append_trace(
                go.Scatter(
                    name=nombre,
                    x=t,
                    y=[p[c - 1] for p in torque][:bici.n_datos - 2],
                    marker_color=color,
                ),
                row=1,
                col=c,
            )

    # velocidad centros de masa
    for c in range(1, 4):
        # for each column
        datos = zip(
            [bici.vel_CM, bici.vel_I, bici.vel_J, bici.vel_K, bici.vel_L],
            [r"$vel_{CM}$", r"$vel_I$", r"$vel_J$", r"$vel_K$", r"$vel_L$"],
            [
                "rgba(255,105,180,1)",
                "rgba(255, 0, 0, 1)",
                "rgba(17, 133, 17, 1)",
                "rgba(0, 0, 255, 1)",
                "rgba(0, 0, 0, 1)",
            ],
        )
        # plot 1 coordinate for each data set
        for torque, nombre, color in datos:
            # print(f"c: {c}")
            fig.append_trace(
                go.Scatter(
                    name=nombre,
                    x=t,
                    y=[p[c - 1] for p in torque][:bici.n_datos - 1],
                    marker_color=color,
                ),
                row=2,
                col=c,
            )
    # aceleracion centros de masa
    for c in range(1, 4):
        # for each column
        datos = zip(
            [bici.acc_I, bici.acc_J, bici.acc_K, bici.acc_L],
            [r"$acc_{CM}$", r"$acc_I$", r"$acc_J$", r"$acc_K$", r"$acc_L$"],
            [
                "rgba(255,105,180,1)",
                "rgba(255, 0, 0, 1)",
                "rgba(17, 133, 17, 1)",
                "rgba(0, 0, 255, 1)",
                "rgba(0, 0, 0, 1)",
            ],
        )
        # plot 1 coordinate for each data set
        for torque, nombre, color in datos:
            # print(f"c: {c}")
            fig.append_trace(
                go.Scatter(
                    name=nombre,
                    x=t,
                    y=[p[c - 1] for p in torque][:bici.n_datos - 0],
                    marker_color=color,
                ),
                row=3,
                col=c,
            )
    # componentes de torque
    datos = zip(
        [
            [p[0] for p in bici.torque_total],
            [p[1] for p in bici.torque_total],
            [p[2] for p in bici.torque_total],
            [npl.norm(p) for p in bici.torque_total],
        ],
        [r"$\tau_x$", r"$\tau_y$", r"$\tau_z$", r"$\tau_{total}$"],
        [
            "rgba(255, 0, 0, 1)",
            "rgba(17, 133, 17, 1)",
            "rgba(0, 0, 255, 1)",
            "rgba(255,105,180,1)",
        ],
    )
    # print(f"torque: {bici.torque_total}")
    # plot 1 coordinate for each data set
    for torque, nombre, color in datos:
        # print(f"c: {c}")
        fig.append_trace(
            go.Scatter(
                name=nombre,
                x=t,
                y=torque[:bici.n_datos - 2],
                marker_color=color,
            ),
            row=2,
            col=4,
        )

    return fig


def graficar_parcial_2():
    """graficar parcial 1"""
    # graph setup
    bici = bici_dict["bici"]
    fig = make_subplots(y_title=r"$\text{Momento de Inercia}\ [kg\cdot m^2]$",
                        x_title=r"tiempo [s]",
                        rows=3,
                        cols=4,
                        shared_xaxes=False,
                        subplot_titles=[
                            r"$i_{11}$",
                            r"$i_{12}$",
                            r"$i_{13}$",
                            "vista superior",
                            r"$i_{21}$",
                            r"$i_{22}$",
                            r"$i_{23}$",
                            "",
                            r"$i_{31}$",
                            r"$i_{32}$",
                            r"$i_{33}$",
                            "",
                        ])
    # sub figures
    # vista aerea bici
    bici.plot_vista_aerea(fig, row=1, col=4, rx=4)
    # tiempo
    t = list(bici.tiempo)
    # componentes de inercia
    for r in range(0, 3):
        for c in range(0, 3):
            # graph all matrix components, each in a separate list
            i_rc = [m[r][c] for m in bici.matriz_inercia_I]
            j_rc = [m[r][c] for m in bici.matriz_inercia_J]
            k_rc = [m[r][c] for m in bici.matriz_inercia_K]
            l_rc = [m[r][c] for m in bici.matriz_inercia_L]
            fig.append_trace(
                go.Scatter(
                    name=f"i_{r+1}{c+1}",
                    x=t,
                    y=i_rc,
                    marker_color="rgba(255, 0, 0, 1)",
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.append_trace(
                go.Scatter(
                    name=f"j_{r+1}{c+1}",
                    x=t,
                    y=j_rc,
                    marker_color="rgba(17, 133, 17, 1)",
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.append_trace(
                go.Scatter(
                    name=f"k_{r+1}{c+1}",
                    x=t,
                    y=k_rc,
                    marker_color="rgba(0, 0, 255, 1)",
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.append_trace(
                go.Scatter(
                    name=f"l_{r+1}{c+1}",
                    x=t,
                    y=l_rc,
                    marker_color="rgba(0, 0, 0, 1)",
                ),
                row=r + 1,
                col=c + 1,
            )
    return fig


def graficar_parcial_1():
    """graficar parcial 1"""
    # graph setup
    bici = bici_dict["bici"]
    # print(f"angulo z marco bicicleta: {bici.B.az}")
    fig = make_subplots(rows=3,
                        cols=3,
                        shared_xaxes=False,
                        subplot_titles=[
                            r"$\text{cordenadas punto interes}$",
                            r"$\text{velocidades punto interes}$",
                            r"$\text{aceleraciones punto interes}$",
                            r"$\omega\text{ rueda trasera y delantera}$",
                            r"$\alpha \text{ rueda trasera y delantera}$",
                            r"$\text{vista superior}$",
                            r"$\omega\text{ de interes}$",
                            r"$\alpha \text{ de interes}$",
                            "",
                        ])
    fig.update_yaxes(selector=dict(type="scatter"),
                     autorangeoptions=dict(include=[0, 1]))
    # sub figures
    # vista aerea bici
    bici.plot_vista_aerea(fig, row=2, col=3, rx=6, ry=4)
    # tiempo
    t = list(bici.tiempo)[:bici.n_datos - 2]
    # 1 - posicion x, y, z de punto de interes
    fig.append_trace(
        go.Scatter(
            name="pos x",
            x=t,
            y=list(bici.punto_de_interes_x)[0:bici.n_datos - 2],
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="pos y",
            x=t,
            y=list(bici.punto_de_interes_y)[0:bici.n_datos - 2],
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="pos z",
            x=t,
            y=list(bici.punto_de_interes_z)[0:bici.n_datos - 2],
        ),
        row=1,
        col=1,
    )
    # 2 - velocidad de x, y, z de punto de interes
    fig.append_trace(
        go.Scatter(
            name="vel x",
            x=t,
            y=list(bici.punto_de_interes_dx)[0:bici.n_datos - 1],
        ),
        row=1,
        col=2,
    )
    fig.append_trace(
        go.Scatter(
            name="vel y",
            x=t,
            y=list(bici.punto_de_interes_dy)[0:bici.n_datos - 1],
        ),
        row=1,
        col=2,
    )
    fig.append_trace(
        go.Scatter(
            name="vel z",
            x=t,
            y=list(bici.punto_de_interes_dz)[0:bici.n_datos - 1],
        ),
        row=1,
        col=2,
    )
    # 2.1 - aceleracion puntos de interes
    fig.append_trace(
        go.Scatter(
            name="acc x",
            x=t,
            y=list(bici.punto_de_interes_ddx)[0:bici.n_datos - 0],
        ),
        row=1,
        col=3,
    )
    fig.append_trace(
        go.Scatter(
            name="acc y",
            x=t,
            y=list(bici.punto_de_interes_ddy)[0:bici.n_datos - 0],
        ),
        row=1,
        col=3,
    )
    fig.append_trace(
        go.Scatter(
            name="acc z",
            x=t,
            y=list(bici.punto_de_interes_ddz)[0:bici.n_datos - 0],
        ),
        row=1,
        col=3,
    )
    # 3 - velocidad angular rueda trasera y delantera
    fig.append_trace(
        go.Scatter(
            name="𝜔 trasera",  # omega
            x=t,
            y=list(bici.omega_rueda_trasera)[0:bici.n_datos - 2],
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="𝜔 delantera",  # omega
            x=t,
            y=list(bici.omega_rueda_delantera)[0:bici.n_datos - 2],
        ),
        row=2,
        col=1,
    )
    # 4 - aceleracion angular rueda trasera y delantera
    fig.append_trace(
        go.Scatter(
            name="𝛼 trasera",  # alpha
            x=t,
            y=list(bici.alpha_rueda_trasera)[0:bici.n_datos],
        ),
        row=2,
        col=2,
    )
    fig.append_trace(
        go.Scatter(
            name="𝛼 delantera",  # alpha
            x=t,
            y=list(bici.alpha_rueda_delantera)[0:bici.n_datos],
        ),
        row=2,
        col=2,
    )
    # 5 - velocidad angular de interes
    fig.append_trace(
        go.Scatter(
            name="𝜔 interes",  # omega
            x=t,
            y=list(bici.omega_de_interes)[0:bici.n_datos - 1],
        ),
        row=3,
        col=1,
    )
    # 6 - aceleracion angular de interes
    fig.append_trace(
        go.Scatter(
            name="𝛼 interes",  # alpha
            x=t,
            y=list(bici.alpha_de_interes)[0:bici.n_datos - 0],
        ),
        row=3,
        col=2,
    )
    return fig


if __name__ == '__main__':
    print("running app")
    app.run(debug=True)  #, suppress_callback_exceptions=False)

# # pruebas
# base = Subspace(
#     x=0,
#     y=0,
#     z=0,
#     ax=0,
#     ay=0,
#     az=0,
#     parent_space=None,
#     nombre="base",
# )
# sub1 = Subspace(
#     x=0,
#     y=0,
#     z=0,
#     ax=pi / 16,
#     ay=0,
#     az=0,
#     parent_space=base,
#     nombre="sub1",
# )
# p1 = Punto(1, 1, 1, sub1, "p1")
# print(f"p1 en base: {p1.llevar_a(base)}")
