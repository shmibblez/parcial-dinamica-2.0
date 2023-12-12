from typing import TypedDict
import numpy as np
import numpy.linalg as npl
from numpy import pi
from scipy.spatial.transform import Rotation as R
from py import Bicycle, Punto


def nivelarParcial3(self: Bicycle):
    """nivela bicicleta (cambia z de subespacio B)"""
    # centro rueda trasera en A
    crt = self.b.llevar_a_parent_space(self.A)
    h = crt.z
    ajuste = self.r - h
    self.B.z += ajuste
    print(
        f"altura luego de ajuste (deberia ser {np.round(self.r,4)}): {self.b.llevar_a_parent_space(self.base).z}"
    )
    # crt ahora en base
    crt = self.b.llevar_a_parent_space(self.base)
    # ahora encontrar y retornar punto de contacto con el piso en B
    # unitario en base hacia abajo
    abajo = Punto(0, 0, -1, self.base, "abajo unitario en B")
    # piso rueda trasera en base
    prt = crt + abajo * self.r
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


def centro_instantaneo_Parcial3(
        self: Bicycle) -> __centro_instantaneo_Parcial3__:
    """encuentra puntos relacionados a centro instantaneo"""
    prt = nivelarParcial3(self)
    # CENTRO INSTANTANEO
    # encontrar velocidad rueda trasera (v = omega * r)
    vrt = self.d_q4 * self.r
    # encontrar diametro de centro instantaneo dado longitud de arco y cambio en angulo
    # en 1 segundo
    d_angulo = np.abs(self.d_q1 * 1)
    l_arco = vrt * 1
    # circumference
    C = (l_arco * 2 * pi) / d_angulo
    # radio centro instantaneo
    rci = C / (2 * pi)
    # direccion centro instantaneo desde origen base en base
    dci = Punto(0, 1 if self.d_q1 < 0 else -1, 0, self.base, "dci")
    # vector de radio centro instantaneo
    vrci = dci * rci
    # ubicacion de centro instantaneo con respecto a base
    ci = vrci + Punto(prt.x, prt.y, 0, self.base, "ajuste ci")
    return {
        "vrci": vrci,
        "ci": ci,
        "d_angulo": d_angulo,
    }


def moverParcial3(self: Bicycle, ):
    """mueve bicicleta si esta parada en una rueda"""
    # primero que todo, encontrar punto de contacto con piso en base, y nivelar bici
    # solo es cuestion de cambiar z de subespacio B
    prt = nivelarParcial3(self)
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
        F_acc = self.m * npl.norm(self.acc_CM[0])
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
        acc = self.acc_CM[-1]
    else:
        acc = [0, 0, 0]
    ## TODO: vectores de aceleracion estan con especto a piso (A)
    ## toca hacer una nueva lista de pos, vel, y acc centro de masa pero
    # con respecto a base
    acc = Punto.from_list(
        acc,
        self.base,
    )

    # CENTRO INSTANTANEO
    datos = centro_instantaneo_Parcial3(self)
    # vector de radio centro instantaneo
    vrci = datos["vrci"]
    # ubicacion de centro instantaneo con respecto a base
    ci = datos["ci"]
    # d_angulo
    d_q1 = datos["d_angulo"]

    # rotar centro instantaneo
    rot = R.from_euler("z", d_q1 if self.d_q1 > 0 else -d_q1, degrees=False)
    nueva_posicion = Punto.from_list(rot.apply(-vrci), self.base,
                                     "nueva pos en base")
    # cambio en posicion
    dpos = nueva_posicion - (-vrci)
    # actualizar pos
    actualizar_pos_parcial3(self, dpos, d_q1)


def actualizar_pos_parcial3(self: Bicycle, dpos: Punto, d_q1: float):
    """actualizar pos bici"""
    self.base.x += dpos.x
    self.base.y += dpos.y
    # si mas grande que 2 pi entonces reducir
    if self.base.az > (2 * pi) or self.base.az < (-2 * pi):
        self.base.az = np.fmod(self.base.az, 2 * pi)
    self.base.az += d_q1
