"""Condition (Psi) of Section 3: growth of the Fock current function on a sector around the illuminated
half-axis. f(xi) = e^{i xi^3/3} g(xi), g = Fock's integral (1/sqrt(pi)) int e^{i xi t}/w(t) dt,
w = sqrt(pi)(Bi + i Ai), contour from infinity*e^{2 pi i/3} through 0 to +infinity (identified with the exact
circle trace of fock.py in fock_integral_check.py). For complex xi the integral cancels over tens of digits;
mpmath with 50-110 digits. Usage: python fock_sector_check.py DIGITS phi,R,side ...   (xi = -R e^{i side phi})
Results of 2026-11-19: results/fock_sector_check.txt"""
import sys, time, mpmath as mp
mp.mp.dps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
e1 = mp.e ** (2j * mp.pi / 3)
w = lambda t: mp.sqrt(mp.pi) * (mp.airybi(t) + 1j * mp.airyai(t))
def g(xi):
    f1 = lambda r: mp.e ** (1j * xi * r * e1) / w(r * e1) * e1
    f2 = lambda r: mp.e ** (1j * xi * r) / w(r)
    pts = [0, 2, 5, 10, 20, 40, 80, 160]
    return (-mp.quad(f1, pts) + mp.quad(f2, pts)) / mp.sqrt(mp.pi)
f = lambda xi: mp.e ** (1j * xi ** 3 / 3) * g(xi)
for a in sys.argv[2:]:
    ang, R, s = [float(x) for x in a.split(",")]
    t0 = time.time(); xi = -R * mp.e ** (1j * s * ang); fv = f(xi)
    print(f"{ang} {R} {s:+.0f} |f/xi|={float(abs(fv/xi)):.6f} |(f-2i xi) xi^2|={float(abs((fv - 2j*xi)*xi**2)):.5f} "
          f"f={mp.nstr(fv,10)} ({time.time()-t0:.1f}s)", flush=True)
