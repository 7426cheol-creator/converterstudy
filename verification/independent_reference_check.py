#!/usr/bin/env python3
"""Independent reference check of the textbook / instruction numbers (standard library only).

The original package's verify_reference.py / verify_expert.py and reference_results.json were not provided,
so this script re-derives every key reference number from the textbook equations with plain `math`/`cmath`.
It never imports the application (`convlab`), so agreement between this script, the app's metrics and the
tests is agreement between independently written code paths.

    python verification/independent_reference_check.py [--json out.json]

Exit code 1 if any check fails.  Known errata (docs/ERRATA.md) are checked against the corrected value and
report the textbook value alongside.
"""

from __future__ import annotations

import cmath
import json
import math
import sys

PI = math.pi
RESULTS: list[dict] = []


def check(lab: str, name: str, value: float, ref: float, tol: float, note: str = "", erratum: str = "") -> None:
    err = abs(value - ref) / abs(ref) if ref != 0 else abs(value)
    ok = err <= tol
    RESULTS.append({"lab": lab, "name": name, "value": value, "ref": ref, "rel_err": err, "tol": tol, "status": "PASS" if ok else "FAIL", "note": note, "erratum": erratum})


# --------------------------------------------------------------------------------------
# FL01 Buck/Boost
# --------------------------------------------------------------------------------------
def fl01():
    Vin, Vo, L, fs, Io = 48.0, 12.0, 20e-6, 500e3, 5.0
    D = Vo / Vin
    dI = (Vin - Vo) * D / (L * fs)
    check("FL01", "buck ripple ΔI [A]", dI, 0.9, 1e-12)
    check("FL01", "buck inductor RMS [A]", math.sqrt(Io**2 + dI**2 / 12), 5.00675, 1e-5)
    check("FL01", "boost RHP zero [kHz]", 160 * 0.5**2 / (2 * PI * 500e-6) / 1e3, 12.73, 5e-4)


# --------------------------------------------------------------------------------------
# FL08 / EX06 DAB (closed forms and an exact piecewise-linear integration written here)
# --------------------------------------------------------------------------------------
def dab_piecewise(V1, V2, L, fs, w1, w2, phi):
    w = 2 * PI * fs

    def b(th, wd, ph):
        psi = (th - ph) % (2 * PI)
        if psi < wd / 2 or psi >= 2 * PI - wd / 2:
            return 1.0
        if PI - wd / 2 <= psi < PI + wd / 2:
            return -1.0
        return 0.0

    edges = {0.0, 2 * PI}
    for wd, ph in ((w1, 0.0), (w2, phi)):
        for e in (ph - wd / 2, ph + wd / 2, ph + PI - wd / 2, ph + PI + wd / 2):
            edges.add(e % (2 * PI))
    E = sorted(edges)
    segs = [(a, c, V1 * b((a + c) / 2, w1, 0.0) - V2 * b((a + c) / 2, w2, phi), V2 * b((a + c) / 2, w2, phi)) for a, c in zip(E[:-1], E[1:]) if c - a > 1e-15]
    cur = [0.0]
    for a, c, vL, _ in segs:
        cur.append(cur[-1] + vL / (w * L) * (c - a))
    mean = sum((c - a) * (cur[k] + cur[k + 1]) / 2 for k, (a, c, _, _) in enumerate(segs)) / (2 * PI)
    cur = [x - mean for x in cur]
    P = sum(v2 * (c - a) * (cur[k] + cur[k + 1]) / 2 for k, (a, c, _, v2) in enumerate(segs)) / (2 * PI)
    I2 = sum((c - a) * (cur[k] ** 2 + cur[k] * cur[k + 1] + cur[k + 1] ** 2) / 3 for k, (a, c, _, _) in enumerate(segs)) / (2 * PI)
    return P, math.sqrt(I2), max(abs(x) for x in cur)


def sps_phi(V1, V2, L, fs, P):
    K = V1 * V2 / (2 * PI * fs * L)
    return PI / 2 * (1 - math.sqrt(1 - 4 * P / (PI * K)))


def fl08_ex06():
    n, V1, L, fs = 50 / 3, 800.0, 200e-6, 100e3
    V2 = n * 48
    phi = sps_phi(V1, V2, L, fs, 1500)
    check("FL08", "SPS φ [rad]", phi, 0.328972794, 1e-8)
    P, I, Ip = dab_piecewise(V1, V2, L, fs, PI, PI, phi)
    check("FL08", "piecewise P [W]", P, 1500.0, 1e-9)
    check("FL08", "L RMS [A]", I, 2.019881509, 1e-8)
    check("FL08", "L peak [A]", Ip, 2.09430585, 1e-8)
    check("FL08", "secondary AC RMS n·I [A]", n * I, 33.6647, 5e-6)
    check("FL08", "Pmax 800/48 [W]", V1 * V2 / (8 * fs * L), 4000.0, 1e-12)
    check("FL08", "Pmax 550/36 [W]", 550 * n * 36 / (8 * fs * L), 2062.5, 1e-12)
    _, I0, Ip0 = dab_piecewise(900, 600, L, fs, PI, PI, 0.0)
    check("FL08", "mismatch φ=0 RMS [A]", I0, 2.165063509, 1e-8)
    check("FL08", "mismatch φ=0 peak [A]", Ip0, 3.75, 1e-12)
    phi6 = sps_phi(900, 600, L, fs, 1500)
    _, I6, Ip6 = dab_piecewise(900, 600, L, fs, PI, PI, phi6)
    check("EX06", "SPS φ 900/600 [rad]", phi6, 0.399994, 5e-6)
    check("EX06", "SPS RMS [A]", I6, 3.113563, 1e-6)
    check("EX06", "SPS peak [A]", Ip6, 5.659830, 1e-6)
    lo, hi = 0.3, 0.8
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if dab_piecewise(900, 600, L, fs, 0.7 * PI, PI, m)[0] < 1500 else (lo, m)
    _, Ic, Ipc = dab_piecewise(900, 600, L, fs, 0.7 * PI, PI, lo)
    check("EX06", "candidate φ_c [rad]", lo, 0.499016363, 1e-8)
    check("EX06", "candidate RMS [A]", Ic, 2.8930917, 1e-7)
    check("EX06", "candidate peak [A]", Ipc, 5.00762763, 1e-7)
    check("EX06", "conduction ratio (I_c/I_sps)²", (Ic / I6) ** 2, 0.8634, 1e-4)
    K = 900 * 600 / (2 * PI * fs * L)
    dphi = 2 * PI * fs * 10e-9
    check("EX06", "timer step dP (linear) [W]", K * (1 - 2 * phi6 / PI) * dphi, 20.1246, 1e-5, erratum="E-001: textbook states 20.62 W")


# --------------------------------------------------------------------------------------
# FL09 LLC and FL10 / EX05 CLLC (FHA written with complex numbers here)
# --------------------------------------------------------------------------------------
def llc_gain(F, k, Q):
    return 1 / abs(1 + (1 - F**-2) / k + 1j * Q * (F - 1 / F))


def cllc(f, n, L1, C1, Lm, L2p, C2p, Vo, Po):
    w = 2 * PI * f
    Rac = 8 / PI**2 * n * n * Vo * Vo / Po
    Z1 = 1j * w * L1 + 1 / (1j * w * C1)
    Z2 = 1j * w * L2p + 1 / (1j * w * C2p)
    Zm = 1j * w * Lm
    Zb = Z2 + Rac
    Zp = Zm * Zb / (Zm + Zb)
    return abs(Zp / (Z1 + Zp) * Rac / Zb), Z1 + Zp


def cllc_scan(n, Vo, Vi, Po, L1=40e-6, C1=28.144773e-9, Lm=200e-6, L2p=None, C2p=None, N=90001):
    L2p = L1 if L2p is None else L2p
    C2p = C1 if C2p is None else C2p
    req = n * Vo / Vi
    best, roots, prev = (-1.0, 0.0), [], None
    for k in range(N):
        f = 120e3 + 90e3 * k / (N - 1)
        g, Zin = cllc(f, n, L1, C1, Lm, L2p, C2p, Vo, Po)
        ind = Zin.imag > 0
        if ind and g > best[0]:
            best = (g, f)
        if prev and (prev[1] - req) * (g - req) < 0 and ind and prev[2]:
            a, b, ga = prev[0], f, prev[1]
            for _ in range(100):
                m = (a + b) / 2
                gm = cllc(m, n, L1, C1, Lm, L2p, C2p, Vo, Po)[0]
                a, b, ga = (m, b, gm) if (ga - req) * (gm - req) > 0 else (a, m, ga)
            roots.append((a + b) / 2)
        prev = (f, g, ind)
    return req, best, roots


def fl09_fl10_ex05():
    check("FL09", "f_r [kHz]", 1 / (2 * PI * math.sqrt(40e-6 * 28.1448e-9)) / 1e3, 150.0, 1e-6)
    check("FL09", "|H| F=1 (any Q)", llc_gain(1.0, 5.0, 1.5), 1.0, 1e-12)
    check("FL09", "n half bridge 400→48 V", 400 / (2 * 48), 4.1666667, 1e-7)
    check("FL09", "n full bridge 400→48 V", 400 / 48, 8.3333333, 1e-7)
    req, best, roots = cllc_scan(1.0, 920, 850, 11000)
    check("FL10", "seed required gain", req, 1.082353, 5e-7)
    check("FL10", "seed max inductive gain", best[0], 1.016401, 5e-7)
    check("FL10", "seed has no root (count)", float(len(roots)), 0.0, 0.0)
    n = 0.93
    check("FL10", "L2 = L1/n² [µH]", 40 / n**2, 46.248121, 1e-8)
    check("FL10", "C2 = C1·n² [nF]", 28.144773 * n**2, 24.342414, 1e-8)
    for Vo, Vi, ref in ((650, 700, [164390.0]), (800, 800, [162811.0]), (920, 850, [136099.47, 147060.86])):
        _, _, r = cllc_scan(n, Vo, Vi, 11000)
        for got, want in zip(r, ref):
            check("FL10", f"n=0.93 root {Vo}/{Vi} V [Hz]", got, want, 1e-5)
    for f0, s_ref, i_ref in ((136099.47, 0.001871, 14.390), (147060.86, -0.001803, 14.765)):
        g1 = cllc(f0 + 10, n, 40e-6, 28.144773e-9, 200e-6, 40e-6, 28.144773e-9, 920, 11000)[0]
        g0 = cllc(f0 - 10, n, 40e-6, 28.144773e-9, 200e-6, 40e-6, 28.144773e-9, 920, 11000)[0]
        check("FL10", f"gain slope at {f0:.0f} Hz [1/kHz]", (g1 - g0) / 20 * 1e3, s_ref, 5e-3)
        _, Zin = cllc(f0, n, 40e-6, 28.144773e-9, 200e-6, 40e-6, 28.144773e-9, 920, 11000)
        check("FL10", f"FHA primary RMS at {f0:.0f} Hz [A]", 2 * math.sqrt(2) / PI * 850 / abs(Zin), i_ref, 1e-4)
    for kl, kc, ref in ((1.05, 1.05, 142.857), (0.95, 0.95, 157.895), (1.05, 0.95, 150.188)):
        check("EX05", f"f_r with L×{kl}, C×{kc} [kHz]", 150 / math.sqrt(kl * kc), ref, 5e-6)
    check("EX12", "grid-limited battery power [kW]", math.sqrt(3) * 400 * 16 * 0.995 * 0.97 / 1e3, 10.6988, 5e-6)


# --------------------------------------------------------------------------------------
# Devices, thermal, magnetics, inverter, PFC, control, system (textbook / instruction numbers)
# --------------------------------------------------------------------------------------
def devices_and_more():
    C0, V0, Vb = 2e-9, 40.0, 800.0
    u = 1 + Vb / V0
    Q = 2 * C0 * V0 * (math.sqrt(u) - 1)
    E = C0 * V0 * V0 * (2 / 3 * u**1.5 - 2 * u**0.5 - (2 / 3 - 2))
    check("EX02", "Qoss(800 V) [nC]", Q * 1e9, 573.212, 1e-6)
    check("EX02", "Eoss(800 V) [µJ]", E * 1e6, 180.238, 3e-6, note="tolerance = half the last printed digit")
    check("EX02", "full transition at 4 A [ns]", 2 * Q / 4 * 1e9, 286.606, 1e-6)
    check("EX02", "single-point C(800 V) estimate [ns]", 2 * C0 / math.sqrt(u) * Vb / 4 * 1e9, 174.574, 3e-6, note="tolerance = half the last printed digit")
    R = [3.6, 4.0, 4.0, 4.4]
    g = [1 / r for r in R]
    for k, ref in enumerate([110.553, 99.497, 99.497, 90.452]):
        check("EX03", f"static branch current {k + 1} [A]", 400 * g[k] / sum(g), ref, 5e-6)

    def e_ov(delta, tr=40.0, V=800.0, I=100.0, N=200000):
        # overlap energy: v falls linearly over [0, tr]; i rises linearly over [delta, delta + tr]
        t0, t1 = -20.0, 80.0
        h = (t1 - t0) / N
        s = 0.0
        for k in range(N):
            t = t0 + (k + 0.5) * h
            v = V if t < 0 else (V * (1 - t / tr) if t < tr else 0.0)
            tt = t - delta
            i = 0.0 if tt < 0 else (I * tt / tr if tt < tr else I)
            s += v * i * h
        return s * 1e-9

    E0 = e_ov(0.0)
    check("EX03", "nominal switching energy [µJ]", E0 * 1e6, 533.333, 1e-5)
    check("EX03", "current channel −5 ns [%]", (e_ov(-5.0) / E0 - 1) * 100, 41.99, 1e-3)
    check("EX03", "current channel +5 ns [%]", (e_ov(5.0) / E0 - 1) * 100, -33.01, 1e-3)
    check("EX04", "window area, rounded currents [mm²]", (50 * 2.02 / 4 + 3 * 33.665 / 4) / 0.30, 168.329, 1e-5)
    check("EX04", "window area, exact currents [mm²]", (50 * 2.019881509 / 4 + 3 * (50 / 3 * 2.019881509) / 4) / 0.30, 168.3234591, 1e-7, erratum="E-002: textbook 168.329 uses rounded currents")
    check("EX04", "harmonic copper loss [W]", 10**2 * 0.02 + 3**2 * 0.04 + 2**2 * 0.06, 2.60, 1e-12)
    Vs, Rs, Ls, P = 400.0, 0.2, 1e-3, 10e3
    Ve = (Vs + math.sqrt(Vs * Vs - 4 * Rs * P)) / 2
    gg = P / Ve**2
    check("EX07", "CPL equilibrium V_e [V]", Ve, 394.935887, 1e-8)
    check("EX07", "incremental resistance [Ω]", -1 / gg, -15.5974, 1e-5)
    check("EX07", "critical C [µF]", Ls * gg / Rs * 1e6, 320.565519, 1e-8)
    for C, re_ref in ((100e-6, 220.57), (1e-3, -67.94)):
        tr = -Rs / Ls + gg / C
        det = (1 - Rs * gg) / (Ls * C)
        check("EX07", f"CPL pole real part C = {C * 1e6:g} µF [1/s]", tr / 2, re_ref, 5e-4)
        check("EX07", f"CPL pole imag part C = {C * 1e6:g} µF [rad/s]", cmath.sqrt(tr * tr / 4 - det).imag, 3134.19 if C < 5e-4 else 991.24, 5e-5)
    check("EX07", "delay 15 µs at 1 kHz [deg]", 360 * 1e3 * 15e-6, 5.4, 1e-12)
    check("EX07", "delay 15 µs at 10 kHz [deg]", 360 * 1e4 * 15e-6, 54.0, 1e-12)
    Z = [[0.2, 0.05], [0.05, 0.25]]
    P0 = [100.0, 80.0]
    a = 0.004
    rise = [Z[0][0] * P0[0] + Z[0][1] * P0[1], Z[1][0] * P0[0] + Z[1][1] * P0[1]]
    M = [[1 - Z[0][0] * a * P0[0], -Z[0][1] * a * P0[1]], [-Z[1][0] * a * P0[0], 1 - Z[1][1] * a * P0[1]]]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    check("EX08", "coupled rise node 1 [K]", (rise[0] * M[1][1] - M[0][1] * rise[1]) / det, 26.569592, 1e-7)
    check("EX08", "coupled rise node 2 [K]", (M[0][0] * rise[1] - M[1][0] * rise[0]) / det, 27.751513, 1e-7)
    B = [[Z[0][0] * a * P0[0], Z[0][1] * a * P0[1]], [Z[1][0] * a * P0[0], Z[1][1] * a * P0[1]]]
    trB = B[0][0] + B[1][1]
    deB = B[0][0] * B[1][1] - B[0][1] * B[1][0]
    check("EX08", "spectral radius", trB / 2 + math.sqrt(trB * trB / 4 - deB), 0.09789, 5e-4)
    check("EX09", "CM peak current [A]", 100e-12 * 50e3 / 1e-6, 5.0, 1e-12)
    check("EX09", "ringing frequency [MHz]", 1 / (2 * PI * math.sqrt(10e-9 * 1e-9)) / 1e6, 50.329, 1e-5)
    check("EX09", "leakage current [mA]", 2 * PI * 50 * 2.2e-9 * 230 * 1e3, 0.158965, 1e-5)
    C, V0f, L, Rf = 1e-3, 800.0, 10e-6, 0.1
    al = Rf / (2 * L)
    wd = math.sqrt(1 / (L * C) - al * al)
    tpk = math.atan(wd / al) / wd
    check("EX10", "stored energy [J]", 0.5 * C * V0f**2, 320.0, 1e-12)
    check("EX10", "time to peak [µs]", tpk * 1e6, 120.919958, 1e-8)
    check("EX10", "peak current [A]", V0f / (L * wd) * math.exp(-al * tpk) * math.sin(wd * tpk), 4370.344, 1e-6)
    Rd = 2 / (C * math.log(800 / 60))
    check("EX10", "discharge resistor [Ω]", Rd, 772.121, 1e-6)
    check("EX10", "initial discharge power [W]", 800**2 / Rd, 828.885, 1e-6)
    check("EX10", "removed energy [J]", 0.5 * C * (800**2 - 60**2), 318.2, 1e-12)
    ui, uo = 11.0, 10.78
    check("EX11", "loss uncertainty, independent [W]", math.sqrt(ui * ui + uo * uo), 15.40157, 1e-6)
    check("EX11", "loss uncertainty, ρ = 0.9 [W]", math.sqrt(ui * ui + uo * uo - 2 * 0.9 * ui * uo), 4.87487, 1e-5)
    check("EX11", "efficiency uncertainty", 0.98 * math.sqrt((ui / 11000) ** 2 + (uo / 10780) ** 2), 0.001386, 5e-4)
    check("EX11", "0/1000 one-sided 95 % bound [%]", (1 - 0.05 ** (1 / 1000)) * 100, 0.2991, 5e-4)


def basics():
    w = 2 * PI * 6000 / 60
    T = 250e3 / w
    iq = T / (1.5 * 4 * (0.115 + (0.15e-3 - 0.35e-3) * (-200)))
    we = 4 * w
    vd = 0.015 * (-200) - we * 0.35e-3 * iq
    vq = 0.015 * iq + we * (0.15e-3 * (-200) + 0.115)
    check("FL04", "shaft torque [N·m]", T, 397.887, 1e-6)
    check("FL04", "iq [A]", iq, 427.835869, 1e-8)
    check("FL04", "|v| required [V]", math.hypot(vd, vq), 438.545444, 1e-8)
    check("FL04", "|v| available (0.95·Vdc/√3) [V]", 800 / math.sqrt(3) * 0.95, 438.786205, 1e-8)
    check("FL04", "phase RMS [A]", math.hypot(200, iq) / math.sqrt(2), 333.949, 1e-6)
    check("FL05", "input current 11 kW [A]", 11000 / (math.sqrt(3) * 400 * 0.97 * 0.995), 16.45043, 1e-6)
    check("FL05", "power at 16 A [W]", math.sqrt(3) * 400 * 16 * 0.97 * 0.995, 10698.81, 1e-6)
    check("FL05", "input current at 360 V [A]", 11000 / (math.sqrt(3) * 360 * 0.97 * 0.995), 18.278, 5e-5)
    check("FL05", "modulation index 400 V / 800 V", 2 * math.sqrt(2) * 400 / math.sqrt(3) / 800, 0.8165, 5e-5)
    check("FL05", "DC-link C single phase [mF]", 7400 / (2 * PI * 50 * 20 * 400) * 1e3, 2.944, 5e-4)
    check("FL05", "hold-up C [µF]", 2 * 11000 * 2e-3 / (800**2 - 760**2) * 1e6, 705.0, 5e-4)
    check("FL06", "Kp [V/A]", 0.8e-3 * 2 * PI * 1e3, 5.026548, 1e-7)
    check("FL06", "Ki [V/(A·s)]", 0.1 * 2 * PI * 1e3, 628.318531, 1e-9)
    check("FL06", "delay phase at 1 kHz [deg]", 360 * 1e3 * 37.5e-6, 13.5, 1e-12)
    check("FL03", "T after one τ [°C]", 65 + 20 * (1 - math.exp(-1)), 77.642411, 1e-8)
    check("FL07", "peak flux density [T]", 1000 / (4 * 100e3 * 50 * 250e-6), 0.2, 1e-12)
    check("FL07", "skin depth Cu 100 kHz [mm]", math.sqrt(1.72e-8 / (PI * 100e3 * 4e-7 * PI)) * 1e3, 0.209, 5e-3)
    check("FL11", "PSFB ideal V_o [V]", 800 / 12 * 0.72, 48.0, 1e-12)
    check("FL11", "12 V path loss vs 48 V (I²R ratio)", (250**2 * 1e-3) / (62.5**2 * 1e-3), 16.0, 1e-12)


def main():
    fl01()
    fl08_ex06()
    fl09_fl10_ex05()
    devices_and_more()
    basics()
    fails = [r for r in RESULTS if r["status"] != "PASS"]
    w = max(len(r["name"]) for r in RESULTS)
    for r in RESULTS:
        extra = f"   [{r['erratum']}]" if r["erratum"] else ""
        print(f"{r['status']:4s} {r['lab']:5s} {r['name']:<{w}s} {r['value']:.10g}  ref {r['ref']:.10g}  rel {r['rel_err']:.1e}{extra}")
    print(f"\n{len(RESULTS)} checks, {len(fails)} failed (standard library only; convlab not imported)")
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"checks": RESULTS, "failed": len(fails), "python": sys.version.split()[0]}, f, ensure_ascii=False, indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
