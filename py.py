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

    def llevar_a_parent_space(self,
                              p: "Punto",
                              target: "Subspace",
                              debug=False) -> "Punto":
        """lleva punto a parent_space s"""
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
        super().append(x)
        super().append(y)
        super().append(z)
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

    def dot(self, arr: "Punto") -> float:
        """dot product between vectors"""
        return np.dot(self, arr)

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


class Bicycle:
    """fuck"""

    def __init__(self, q1, q2, q3, q4, q5, q6, q7, d_q4, dt):
        self.r = 0.25
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
        self.d_q4 = d_q4  # velocidad angular rueda trasera
        # variables
        self.t = 0  # tiempo actual
        self.dt = dt  # intervalo de tiempo
        # listas de datos
        self.n_datos = 42  # para que sean 40 luego de doble derivada
        self.tiempo = deque(maxlen=self.n_datos)
        # 1 - posicion x, y, z de punto de interes
        self.punto_de_interes_x = deque(maxlen=self.n_datos)
        self.punto_de_interes_y = deque(maxlen=self.n_datos)
        self.punto_de_interes_z = deque(maxlen=self.n_datos)
        # 2 - velocidad de x, y, z de punto de interes
        self.punto_de_interes_dx = deque(maxlen=self.n_datos)
        self.punto_de_interes_dy = deque(maxlen=self.n_datos)
        self.punto_de_interes_dz = deque(maxlen=self.n_datos)
        # 3 - velocidad angular rueda trasera y delantera
        self.omega_rueda_delantera = deque(maxlen=self.n_datos)
        self.omega_rueda_trasera = deque(maxlen=self.n_datos)
        # 4 - aceleracion angular rueda trasera y delantera
        self.alpha_rueda_delantera = deque(maxlen=self.n_datos)
        self.alpha_rueda_trasera = deque(maxlen=self.n_datos)
        # 5 - velocidad angular de interes
        self.angulo_de_interes = deque(maxlen=self.n_datos)
        self.omega_de_interes = []
        # 6 - aceleracion angular de interes
        self.alpha_de_interes = []

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
        self.B = Subspace(
            x=0,
            y=0,
            z=0,
            ax=0,
            ay=0,
            az=self.q1,  # yaw bici
            parent_space=self.A,
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
        self.b = Punto(0, 0, 0, self.E, "b - centro rueda trasera")
        self.c = Punto(0, 0, 0, self.G, "c - centro superior manuvrio")
        self.d = Punto(0, 0, 0, self.H, "d - centro rueda delantera")

    def mover(self):
        """mueve bici un paso"""
        for _ in range(1, 6):
            self.adjust_pitch()  # adjust multiple times

        # encontrar velocidad linear de rueda trasera
        v_rueda_trasera = self.d_q4 / self.r
        omega_rueda_trasera = v_rueda_trasera * self.r
        # omega_rueda_delantera = 0

        if self.q6 == 0:
            # si angulo manuvrio == 0
            # centro instantaneo infinito
            omega_rueda_delantera = omega_rueda_trasera
            # se mueve en eje x de B unicamente
            self.update_pos_B(v_rueda_trasera * self.dt, 0, 0)
        else:
            # si angulo manuvrio != 0
            # centro instantaneo existe
            # encontrar centro instantaneo en B
            resp = self.encontrar_centro_instantaneo()
            # centro instantaneo
            ci_B = resp["centro_instantaneo_proyectado"]
            # crd_B = resp["centro_de_rueda_delantera_proyectado"]
            prd_B = resp["punto_de_contacto_rueda_delantera_proyectado"]
            crt_B = resp["centro_de_rueda_trasera_proyectado"]
            # distancia entre centro instantaneo y punto de
            # contacto rueda delantera
            r_delantero_piso = npl.norm(ci_B - prd_B)
            # distancia entre centro instantaneo y punto de
            # contacto rueda trasera
            r_trasera = npl.norm(ci_B - crt_B)
            # velocidad angular centro instantaneo
            omega_ci = v_rueda_trasera * r_trasera
            # calcular velocidad rueda delatera
            v_rueda_delantera = omega_ci / r_delantero_piso
            # calcular omega rueda delantera
            omega_rueda_delantera = v_rueda_delantera * self.r
            # TODO: actualizar posicion, encontrar
            # TODO: vector de interes para eso
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
            # print(
            #     f"a_rotacion: {np.round(a_rotacion,2)}, dpos: {np.round(dpos,2)}, B: {np.round(self.B.origin.llevar_a_parent_space(self.A),2)}, q1: {self.q1}"
            # )

        # agregar valores a lista
        self.tiempo.append(self.update_time())
        # 1 - posicion x, y, z de punto de interes
        # 2 - velocidad de x, y, z de punto de interes
        punto_interes = self.B.origin.llevar_a_parent_space(
            self.A)  #, debug=True)
        self.update_punto_interes(punto_interes)
        # 3 - velocidad angular rueda trasera y delantera
        # 4 - aceleracion angular rueda trasera y delantera
        self.update_datos_ruedas(omega_rueda_trasera, omega_rueda_delantera)
        # 5 - velocidad angular de interes
        # 6 - aceleracion angular de interes
        angulo_interes = self.B.az
        self.update_angulos_interes(angulo_interes)

    class __encontrar_centro_instantaneo(TypedDict):
        """tipo de retorno de encontrar_centro_instantaneo()"""
        centro_de_rueda_trasera_proyectado: Punto
        centro_de_rueda_delantera_proyectado: Punto
        punto_de_contacto_rueda_delantera_proyectado: Punto
        centro_instantaneo_proyectado: Punto

    def encontrar_centro_instantaneo(self) -> __encontrar_centro_instantaneo:
        """retorna centro instantaneo en B"""
        # d2 en D es perpendicular a rueda trasera
        # tambien esta en plano xy de D en D
        v1_D = Punto(0, 1, 0, self.D, "d2_D")
        # h2 en H es perpendicular a rueda delantera
        OH_H = Punto(0, 0, 0, self.H, "OH_H")
        # si volteando a izquierda apuntar a izquierda
        # si volteando a derecha apuntar a derecha
        h2_H = Punto(0, -1 if self.q6 > 0 else 1, 0, self.H, "h2_H")
        # llevar a B
        OH_D = OH_H.llevar_a_parent_space(self.D)
        h2_D = h2_H.llevar_a_parent_space(self.D) - OH_D
        # proyectar en plano xy de D en D
        v2_D = h2_D.project_onto_plane(Punto(0, 0, 1, self.D, "d3_D"),
                                       parent_space=self.D)
        # si h2 apunta en direccion opuesta, voltear
        # componente en x deberia ser negativo
        if v2_D.x > 0:
            v2_D.x = -v2_D.x
            v2_D.y = -v2_D.y
            v2_D.z = -v2_D.z
        # encontrar angulo entre b1_B y v2_B
        d1_D = Punto(1, 0, 0, self.D, "d1_D")
        a1 = d1_D.angulo_entre(v2_D)
        # encontrar angulo entre vectores
        # a2 = v1_B.angulo_entre(v2_B)
        # distancia entre puntos b y d
        d = npl.norm(
            self.b.llevar_a_parent_space(self.D) -
            self.d.llevar_a_parent_space(self.D))
        # encontrar longitud de v1_B y v2_B
        norm_v1 = np.tan(a1) * d
        norm_v2 = d / np.cos(a1)
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
        v3_D = self.b.llevar_a_parent_space(
            self.D) - self.d.llevar_a_parent_space(self.D)
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
        v5_B = v4_B + v2_B
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
        v9_B = v8_B + v4_B
        # retornar
        return {
            "centro_de_rueda_trasera_proyectado": v4_B,
            "centro_de_rueda_delantera_proyectado": v5_B,
            "punto_de_contacto_rueda_delantera_proyectado": v9_B,
            "centro_instantaneo_proyectado": v6_B,
        }

    def adjust_pitch(self):
        """
        adjusts pitch so front wheel touches ground
        needs to be called at least 4 times
        """
        # vectore normal a plano xz de H en H
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
        if v3_B.z > 0:
            ajuste = -ajuste
        self.set_q3(self.q3 + ajuste)
        # print(f"ajuste: {ajuste}")

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
        # print(f"q1: {self.q1}")

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
        # derivadas de cordenadas
        self.punto_de_interes_dx = np.diff(self.punto_de_interes_x)
        self.punto_de_interes_dy = np.diff(self.punto_de_interes_y)
        self.punto_de_interes_dz = np.diff(self.punto_de_interes_z)

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
    def update_datos_ruedas(self, omega_trasera, omega_delantera):
        """actualiza datos de rueda trasera y delantera"""
        # velocidades angulares
        self.omega_rueda_trasera.append(omega_delantera)
        self.omega_rueda_delantera.append(omega_trasera)
        # aceleraciones
        self.alpha_rueda_trasera = np.divide(
            Bicycle.__diff_angulos__(self.omega_rueda_trasera), self.dt)
        self.alpha_rueda_delantera = np.divide(
            Bicycle.__diff_angulos__(self.omega_rueda_delantera), self.dt)

    # 5 - velocidad angular de interes
    # 6 - aceleracion angular de interes
    def update_angulos_interes(self, angulo_interes: float):
        """actualiza angulo de interes, y su primera y segunda derivada"""
        # print(f"angulo interes: {angulo_interes}")
        self.angulo_de_interes.append(angulo_interes)
        self.omega_de_interes = np.divide(
            Bicycle.__diff_angulos__(self.angulo_de_interes), self.dt)
        self.alpha_de_interes = np.divide(
            Bicycle.__diff_angulos__(self.omega_de_interes, debug=True),
            self.dt)


# crear bici
bici_dict = {
    "bici":
    Bicycle(
        q1=0,  # bike yaw
        q2=pi / 4,  # bike roll
        q3=0,  #pi / 8,  # bike pitch
        q4=0,  # back wheel angle
        q5=0,  #pi / 8,  # fork pitch angle en rango [0, -pi/4]
        q6=-pi / 4,  # fork yaw
        q7=0,  # front wheel angle
        d_q4=pi / 4,  # back wheel angular speed
        dt=0.25,  # time interval
    )
}
print("bici creada")

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='grafica',
              responsive=True,
              style={
                  "width": "90vh",
                  "height": "90vh"
              }),
    dcc.Interval(
        id="actualizar-bici",
        interval=750,  # ms
        n_intervals=0,
    )
])


@callback(Output("grafica", "figure"), Input("actualizar-bici", "n_intervals"))
def update_graph(_):
    """update graph"""
    bici = bici_dict["bici"]
    bici.mover()
    # graph setup
    fig = make_subplots(rows=3,
                        cols=2,
                        shared_xaxes=True,
                        subplot_titles=[
                            "cordenadas punto interes",
                            "velocidades punto interes",
                            "𝜔 rueda trasera y delantera",
                            "𝛼 rueda trasera y delantera",
                            "𝜔 de interes",
                            "𝛼 de interes",
                        ])
    # sub figures
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
            y=list(bici.alpha_rueda_trasera)[0:bici.n_datos - 1],
        ),
        row=2,
        col=2,
    )
    fig.append_trace(
        go.Scatter(
            name="𝛼 delantera",  # alpha
            x=t,
            y=list(bici.alpha_rueda_delantera)[0:bici.n_datos - 1],
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
