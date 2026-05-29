"""
Modelo base:
  · Llegadas NSPP (Poisson No Homogéneo) según distribución horaria real
  · Entidades: vehículos chico/mediano/grande, combustible super/premium/gasoil
  · 6 islas de carga; playeros: 3 en turno día, 1 en turno noche
  · Cambio de turno: 10 min de bloqueo (sin nuevos ingresos a islas)
  · Abandono: tolerancia = UMBRAL + Exp(pendiente)
  · Empleado inexperto: tiempo de pago × 2 (primeras 2 semanas)
  · 30 corridas independientes con IC 95% por escenario
"""

import simpy
import random
import numpy as np

# ================================================================
#  PARÁMETROS – modificar aquí para distintos escenarios
# ================================================================

DURACION_SIM       = 24 * 60      # minutos simulados (1 día)
NUM_CORRIDAS       = 30           # corridas por escenario

# Recursos fijos
NUM_ISLAS          = 6
PLAYEROS_DIA       = 3            # turno mañana (06-14) y tarde (14-22)
PLAYEROS_NOCHE     = 1            # turno noche (22-06)

# Demanda
DEMANDA_SEMANA     = 2300         # veh/día hábil
DEMANDA_FINDE      = 3200         # veh/día fin de semana

# Distribución horaria (proporción de llegadas por hora, suma = 1)
DIST_HORARIA = [
    0.0129, 0.0082, 0.0054, 0.0046, 0.0054, 0.0098,   # 00-05 h
    0.0169, 0.0322, 0.0438, 0.0438, 0.0480, 0.0542,   # 06-11 h
    0.0610, 0.0664, 0.0666, 0.0694, 0.0757, 0.0812,   # 12-17 h
    0.0819, 0.0679, 0.0521, 0.0411, 0.0309, 0.0207,   # 18-23 h
]

# Caudales de surtidor por tipo de combustible (L/min)
CAUDALES = {"super": 40, "premium": 38, "gasoil": 50}


# Turnos: (minuto_del_día_en_que_comienza, playeros_del_nuevo_turno)
# El día arranca a las 00:00 con turno noche (1 playero); a las 06:00 empieza mañana
CAMBIOS_TURNO = [
    ( 6 * 60, PLAYEROS_DIA),      # 06:00 → turno mañana
    (14 * 60, PLAYEROS_DIA),      # 14:00 → turno tarde
    (22 * 60, PLAYEROS_NOCHE),    # 22:00 → turno noche
]
DUR_CAMBIO_TURNO = 10             # minutos de bloqueo durante el relevo

# Abandono: el vehículo espera como máximo UMBRAL + Exp(1/PENDIENTE) minutos
UMBRAL_ABANDONO    = 15.0         # min sin riesgo de abandono
PENDIENTE_ABANDONO = 0.05         # 1/media de la cola exponencial (media = 20 min extra)

# Nivel de servicio objetivo
OBJETIVO_ESPERA    = 15.0         # min


# ================================================================
#  ENTIDAD: VEHÍCULO
# ================================================================

class Vehiculo:
    """
    Atributos asignados al arribar:
      · tipo (chico/mediano/grande) según patentamientos AR 2022-2024
      · combustible y litros → tiempo de carga
    """

    PROB_TIPO = [(0.44, "chico"), (0.71, "mediano"), (1.00, "grande")]
    PROB_COMB = [(0.50, "super"), (0.75, "premium"), (1.00, "gasoil")]

    def __init__(self, vid: int, t_llegada: float, hay_inexperto: bool = False):
        self.vid           = vid
        self.t_llegada     = t_llegada
        self.hay_inexperto = hay_inexperto

        self.tipo        = self.asignar_tipo(self.PROB_TIPO)
        self.combustible = self.asignar_tipo(self.PROB_COMB)
        self.litros      = self.asignar_litros()
        self.t_carga     = self._calcular_t_carga()

    @staticmethod
    def asignar_tipo(tabla):
        r = random.random()
        for umbral, valor in tabla:
            if r < umbral:
                return valor

    def asignar_litros(self) -> float:
        rangos = {"chico": (15, 40, 25), "mediano": (25, 55, 40), "grande": (40, 75, 60)}
        lo, hi, moda = rangos[self.tipo]
        return random.triangular(lo, hi, moda)

    def _calcular_t_carga(self) -> float:
        return self.litros / CAUDALES[self.combustible]

    def t_pago(self) -> float:
        """
        Distribución discreta del tiempo de cobro:
          25 % → 1 min (pago rápido)
          60 % → 2 min (pago normal)
          15 % → 4 min (inconvenientes)
        Si el playero es inexperto, todos los tiempos se duplican.
        """
        r = random.random()
        if r < 0.25:   base = 1.0
        elif r < 0.85: base = 2.0
        else:          base = 4.0
        return base * (2.0 if self.hay_inexperto else 1.0)

    def t_tolerado(self) -> float:
        """
        Tiempo máximo de espera antes de abandonar.
        Modelo: sin riesgo hasta UMBRAL_ABANDONO; luego Exp(PENDIENTE_ABANDONO).
        """
        return UMBRAL_ABANDONO + random.expovariate(PENDIENTE_ABANDONO)


# ================================================================
#  TASA DE LLEGADA NSPP
# ================================================================

def tasa_llegada(t_sim: float, demanda_diaria: float) -> float:
    """Devuelve la tasa instantánea en veh/min según la franja horaria."""
    hora = int((t_sim // 60) % 24)
    return demanda_diaria * DIST_HORARIA[hora] / 60.0


# ================================================================
#  PROCESO DE VEHÍCULO
# ================================================================

def proceso_vehiculo(env, v: Vehiculo, islas, playeros, R: dict):
    """
    Flujo según diagrama lógico del informe:
      1. Solicitar isla  ← con posibilidad de abandono por timeout
      2. Solicitar playero para iniciar atención
      3. Playero libera → la carga transcurre de forma autónoma
      4. Solicitar playero para cobro
      5. Cobrar → liberar isla
    """
    t0 = v.t_llegada

    # ── 1. ESPERAR ISLA (abandono por timeout) ─────────────────
    req_isla   = islas.request()
    t_max_esp  = v.t_tolerado()

    resultado = yield req_isla | env.timeout(t_max_esp)

    if req_isla not in resultado:
        req_isla.cancel()                          # retirar de la cola
        R["abandonos"] += 1
        R["t_esp_aban"].append(env.now - t0)
        return

    # ── 2. ESPERAR PLAYERO PARA INICIO ─────────────────────────
    req_play = playeros.request()
    yield req_play

    espera = env.now - t0
    R["esperas"].append(espera)

    # El playero conecta la manguera y queda disponible
    playeros.release(req_play)

    # ── 3. CARGA AUTÓNOMA (isla ocupada) ───────────────────────
    yield env.timeout(v.t_carga)

    # ── 4. ESPERAR PLAYERO PARA COBRO ──────────────────────────
    req_cobro = playeros.request()
    yield req_cobro
    yield env.timeout(v.t_pago())
    playeros.release(req_cobro)

    # ── 5. LIBERAR ISLA ────────────────────────────────────────
    islas.release(req_isla)

    # ── MÉTRICAS ───────────────────────────────────────────────
    t_sis = env.now - t0
    R["t_sistema"].append(t_sis)
    R["atendidos"]    += 1
    R["ev_atendidos"] += int(v.es_ev)
    if espera <= OBJETIVO_ESPERA:
        R["dentro_obj"] += 1


# ================================================================
#  GENERADOR DE LLEGADAS (NSPP – exponencial por franja horaria)
# ================================================================

def generar_llegadas(env, islas, playeros, estado: dict, R: dict, demanda: float):
    vid = 0
    while True:
        tasa = tasa_llegada(env.now, demanda)
        dt   = random.expovariate(tasa) if tasa > 1e-9 else 60.0
        yield env.timeout(dt)

        if env.now > DURACION_SIM:
            break

        vid += 1
        R["generados"] += 1
        v = Vehiculo(vid, env.now, estado["hay_inexperto"])
        env.process(proceso_vehiculo(env, v, islas, playeros, R))


# ================================================================
#  GESTOR DE TURNOS
# ================================================================

def gestor_turnos(env, playeros_res, estado: dict):
    """
    En cada cambio de turno:
      a) Pone la capacidad de playeros en 0 → nadie puede iniciar atención
         (los vehículos en cola esperan; los que ya están en isla continúan)
      b) Espera DUR_CAMBIO_TURNO minutos
      c) Restablece la capacidad con el nuevo plantel y libera la cola
    """
    for t_cambio, n_play_nuevo in CAMBIOS_TURNO:
        espera = t_cambio - env.now
        if espera > 0:
            yield env.timeout(espera)
        if env.now > DURACION_SIM:
            break

        estado["cambio_turno"] = True
        playeros_res._capacity  = 0          # bloquear nuevas atenciones
        yield env.timeout(DUR_CAMBIO_TURNO)

        playeros_res._capacity  = n_play_nuevo
        estado["cambio_turno"]  = False
        # Notificar a simpy que hay slots disponibles para la cola pendiente
        playeros_res._trigger_put(None)


# ================================================================
#  MONITOR (muestras cada minuto)
# ================================================================

def monitor(env, islas, playeros, R: dict, intervalo: float = 1.0):
    while True:
        yield env.timeout(intervalo)
        if env.now > DURACION_SIM:
            break
        cap = playeros._capacity if playeros._capacity > 0 else 1
        R["m_cola_isla"].append(len(islas.queue))
        R["m_uso_isla"].append(islas.count / NUM_ISLAS)
        R["m_uso_play"].append(playeros.count / cap)


# ================================================================
#  UNA CORRIDA
# ================================================================

def una_corrida(demanda: float, hay_inexperto: bool = False, seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)

    env      = simpy.Environment()
    # El día arranca a las 00:00, que es mitad del turno noche → 1 playero
    islas    = simpy.Resource(env, capacity=NUM_ISLAS)
    playeros = simpy.Resource(env, capacity=PLAYEROS_NOCHE)

    estado = {"cambio_turno": False, "hay_inexperto": hay_inexperto}

    R = {
        "generados"  : 0,
        "atendidos"  : 0,
        "abandonos"  : 0,
        "ev_atendidos": 0,
        "dentro_obj" : 0,
        "esperas"    : [],
        "t_sistema"  : [],
        "t_esp_aban" : [],
        "m_cola_isla": [],
        "m_uso_isla" : [],
        "m_uso_play" : [],
    }

    env.process(gestor_turnos(env, playeros, estado))
    env.process(monitor(env, islas, playeros, R))
    env.process(generar_llegadas(env, islas, playeros, estado, R, demanda))

    # Buffer: deja terminar atenciones en curso hasta 2 h después del cierre
    env.run(until=DURACION_SIM + 120)
    return R


# ================================================================
#  AGREGADO DE CORRIDAS Y ESTADÍSTICAS
# ================================================================

def ic95(arr):
    """(media, IC_inf, IC_sup) con t de Student para n=30 (z≈1.96)."""
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    mu  = a.mean()
    if n < 2:
        return float(mu), float(mu), float(mu)
    err = 1.96 * a.std(ddof=1) / np.sqrt(n)
    return float(mu), float(mu - err), float(mu + err)


def correr_escenario(nombre: str, demanda: float,
                     hay_inexperto: bool = False,
                     n: int = NUM_CORRIDAS) -> dict:
    """Corre n réplicas y muestra la tabla de resultados con IC 95%."""

    keys = ["gen","ate","aba","ev","esp","sis","niv","tab","cola","uisl","uplay"]
    acum = {k: [] for k in keys}

    for seed in range(n):
        R = una_corrida(demanda, hay_inexperto, seed=seed)

        acum["gen"].append(R["generados"])
        acum["ate"].append(R["atendidos"])
        acum["aba"].append(R["abandonos"])
        acum["ev"].append(R["ev_atendidos"])
        acum["esp"].append(np.mean(R["esperas"])   if R["esperas"]    else 0.0)
        acum["sis"].append(np.mean(R["t_sistema"]) if R["t_sistema"]  else 0.0)
        acum["niv"].append(
            R["dentro_obj"] / R["atendidos"] * 100 if R["atendidos"] else 0.0)
        acum["tab"].append(
            R["abandonos"]  / R["generados"]  * 100 if R["generados"] else 0.0)
        acum["cola"].append(np.mean(R["m_cola_isla"]) if R["m_cola_isla"] else 0.0)
        acum["uisl"].append(
            np.mean(R["m_uso_isla"]) * 100 if R["m_uso_isla"] else 0.0)
        acum["uplay"].append(
            np.mean(R["m_uso_play"]) * 100 if R["m_uso_play"] else 0.0)

    # ── Tabla de resultados ─────────────────────────────────────
    SEP = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  ESCENARIO : {nombre}")
    print(f"  Demanda   : {demanda:.0f} veh/día  |  Inexperto: {hay_inexperto}")
    print(f"  Corridas  : {n}")
    print(f"{'═'*60}")
    print(f"\n  {'MÉTRICA':<35} {'MEDIA':>8}    IC 95%")
    print(f"  {SEP}")

    def fila(label, key, fmt, ud=""):
        mu, lo, hi = ic95(acum[key])
        print(f"  {label:<35} {mu:{fmt}}{ud}   [{lo:{fmt}} – {hi:{fmt}}]")

    fila("Vehículos generados",         "gen",  "7.1f")
    fila("Vehículos atendidos",         "ate",  "7.1f")
    fila("Abandonos",                   "aba",  "7.1f")
    fila("EV atendidos",                "ev",   "7.1f")
    print(f"  {SEP}")
    fila("Espera promedio en cola",     "esp",  "7.3f", " min")
    fila("Tiempo promedio en sistema",  "sis",  "7.3f", " min")
    fila("Longitud promedio de cola",   "cola", "7.2f", " veh")
    print(f"  {SEP}")
    fila("Utilización de islas",        "uisl", "7.1f", "%")
    fila("Utilización de playeros",     "uplay","7.1f", "%")
    print(f"  {SEP}")
    fila("Nivel de servicio (<15 min)", "niv",  "7.1f", "%")
    fila("Tasa de abandono",            "tab",  "7.2f", "%")
    print()

    return {k: ic95(v)[0] for k, v in acum.items()}


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  SIMULACIÓN – ESTACIÓN DE SERVICIO RÍO SECO")
    print(f"  Islas: {NUM_ISLAS}  |  Playeros día: {PLAYEROS_DIA}  |  Noche: {PLAYEROS_NOCHE}")
    print(f"|  Corridas: {NUM_CORRIDAS}")
    print(f"  Abandono: umbral {UMBRAL_ABANDONO:.0f} min + Exp(media {1/PENDIENTE_ABANDONO:.0f} min)")
    print("═" * 60)

    # ── Escenario base ──────────────────────────────────────────
    correr_escenario("Base – Día hábil",
                     DEMANDA_SEMANA, hay_inexperto=False)

    correr_escenario("Base – Fin de semana",
                     DEMANDA_FINDE,  hay_inexperto=False)

    # ── Con empleado inexperto ──────────────────────────────────
    correr_escenario("Inexperto – Día hábil",
                     DEMANDA_SEMANA, hay_inexperto=True)

    correr_escenario("Inexperto – Fin de semana",
                     DEMANDA_FINDE,  hay_inexperto=True)