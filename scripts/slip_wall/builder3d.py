"""Task 26 stage 3 (DRAFT, not executed): FP64 PRODUCTION builder of the genuine 3D terrain (STAGE3_DESIGN_NOTE sections 1-3a,
#541, #544). Cartesian x-space BSP of convex polyhedra; independent of the exact oracle (it never imports oracle3d,
exact3d or manifest3d). Inputs: the REGISTERED stage-2 manifest JSON (node heights, diagonals, Z_top).

Rules (design note section 3):
  - a vertex with |n.x - d| <= EPS_ON * max(1, |d|) (n unit) is ON the plane and is snapped onto it; a side is EMPTY
    exactly when no vertex lies strictly on it: a classification decision, never a volume test
  - no piece is ever dropped for its volume; pieces with volume <= SMALL are retained and counted
  - every split records |V(P) - V(P-) - V(P+)| (split defect): max and sum are reported and gated by the stage
  - row existence and orientation: the EXACT sign of c . n_raw from the manifest node values (Fractions; #532)
Products per case: cells (V, centroid, pieces), facets (area vector FP64, plane), rows (facet, d) with vi = c . a,
h[(row, cell, image)] (immediate volume; G and R use it), W[(producer row, receiver facet, image)], direct[(cell, d)],
the sampler, and the diagnostics.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np

EPS_ON = 1e-12
SMALL = 1e-14
LEX19 = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x * x + y * y + z * z <= 2]
OPP19 = [LEX19.index((-c[0], -c[1], -c[2])) for c in LEX19]


class GeometryRejected(Exception):
    pass


class GeometryInvalid(Exception):
    """A non-finite or negative amount, or an exact-positive row with FP64 vi <= 0 (#549): the build is INVALID."""


def checked_volume(p, label):
    v = p.volume()
    if not math.isfinite(v) or v < 0.0:
        raise GeometryInvalid(f"non-finite or negative volume {v} at {label}")
    return v


def exact_parallel(n1, n2):
    return (n1[1] * n2[2] - n1[2] * n2[1], n1[2] * n2[0] - n1[0] * n2[2], n1[0] * n2[1] - n1[1] * n2[0]) == (0, 0, 0)


def exact_sigma_difference(g1, g2, c):
    """sign of sigma_1 - sigma_2 for parallel planes n.x = d: sigma_k(x) = (n_k.x - d_k)/(n_k.c); with n_2 = s n_1 the
    difference is the constant (d_2/(n_2.c) - d_1/(n_1.c)) (exact)."""
    n1, n2 = g1.n_raw_exact, g2.n_raw_exact
    c1 = sum(F(c[k]) * n1[k] for k in range(3))
    c2 = sum(F(c[k]) * n2[k] for k in range(3))
    diff = g2.d_raw_exact / c2 - g1.d_raw_exact / c1
    return (diff > 0) - (diff < 0)


def dotf(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def crossf(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def subf(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class Diag:
    def __init__(self):
        self.max_defect = 0.0
        self.sum_defect = 0.0
        self.small_count = 0
        self.small_total = 0.0
        self.max_unit_sum = 0.0                       # largest split-defect sum of one row or one cell (design note 3)

    def as_dict(self):
        return {"max_split_defect": self.max_defect, "sum_split_defect": self.sum_defect, "max_row_or_cell_defect_sum": self.max_unit_sum,
                "small_pieces": self.small_count,
                "small_total": self.small_total}


class Poly:
    """FP64 convex polyhedron: vertices and face cycles."""

    def __init__(self, v, f):
        self.v, self.f = list(v), [list(x) for x in f]

    @staticmethod
    def box(lo, hi):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        return Poly(v, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]])

    @staticmethod
    def prism(polygon, vec):
        k = len(polygon)
        v = list(polygon) + [(p[0] + vec[0], p[1] + vec[1], p[2] + vec[2]) for p in polygon]
        faces = [list(range(k)), list(range(k, 2 * k))] + [[q, (q + 1) % k, k + (q + 1) % k, k + q] for q in range(k)]
        return Poly(v, faces)

    def interior(self):
        n = len(self.v)
        return (sum(p[0] for p in self.v) / n, sum(p[1] for p in self.v) / n, sum(p[2] for p in self.v) / n)

    def volume(self):
        g = self.interior()
        tot = 0.0
        for face in self.f:
            a = self.v[face[0]]
            for q in range(1, len(face) - 1):
                b, c = self.v[face[q]], self.v[face[q + 1]]
                tot += abs(dotf(subf(a, g), crossf(subf(b, g), subf(c, g))))
        return tot / 6.0

    def moment(self):
        g = self.interior()
        vol, m = 0.0, [0.0, 0.0, 0.0]
        for face in self.f:
            a = self.v[face[0]]
            for q in range(1, len(face) - 1):
                b, c = self.v[face[q]], self.v[face[q + 1]]
                w = abs(dotf(subf(a, g), crossf(subf(b, g), subf(c, g)))) / 6.0
                vol += w
                for k in range(3):
                    m[k] += w * (g[k] + a[k] + b[k] + c[k]) / 4.0
        return vol, m

    def halfspaces(self):
        """Outward face planes (unit n, d) of this convex polyhedron."""
        g = self.interior()
        out = []
        for face in self.f:
            p0 = self.v[face[0]]
            n = (0.0, 0.0, 0.0)
            for q in range(1, len(face) - 1):
                n = tuple(n[k] + crossf(subf(self.v[face[q]], p0), subf(self.v[face[q + 1]], p0))[k] for k in range(3))
            ln = math.sqrt(dotf(n, n))
            if ln == 0.0:
                continue
            n = (n[0] / ln, n[1] / ln, n[2] / ln)
            if dotf(n, subf(g, p0)) > 0:
                n = (-n[0], -n[1], -n[2])
            out.append((n, dotf(n, p0)))
        return out


def clip(P, n, d):
    """P ∩ {n.x <= d} with classification (EPS_ON) and snapping; None when no vertex lies strictly inside."""
    tol = EPS_ON * max(1.0, abs(d))
    val = [dotf(n, p) - d for p in P.v]
    cls = [0 if abs(x) <= tol else (1 if x > 0 else -1) for x in val]
    if all(c >= 0 for c in cls):
        return None
    if all(c <= 0 for c in cls):
        return P
    new_v, index, edge = [], {}, {}

    def keep(i):
        if i not in index:
            p = P.v[i]
            if cls[i] == 0:                                # snap ON vertices onto the plane
                p = (p[0] - val[i] * n[0], p[1] - val[i] * n[1], p[2] - val[i] * n[2])
            index[i] = len(new_v)
            new_v.append(p)
        return index[i]

    def cut(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge:
            s = val[i] / (val[i] - val[j])
            a, b = P.v[i], P.v[j]
            edge[key] = len(new_v)
            new_v.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])))
        return edge[key]
    faces, on_sets = [], []
    for face in P.f:
        out = []
        m = len(face)
        for q in range(m):
            i, j = face[q], face[(q + 1) % m]
            if cls[i] <= 0:
                out.append(keep(i))
            if cls[i] * cls[j] < 0:
                out.append(cut(i, j))
        dd = []
        for x in out:
            if not dd or dd[-1] != x:
                dd.append(x)
        while len(dd) > 1 and dd[0] == dd[-1]:
            dd.pop()
        if len(dd) >= 3:
            faces.append(dd)
        on = [x for x in dd if (x in edge.values()) or any(index.get(i) == x and cls[i] == 0 for i in index)]
        if len(on) >= 2:
            on_sets.append(on)
    cap = sorted({x for s in on_sets for x in s})
    if len(cap) >= 3:
        faces.append(order_cycle(new_v, cap, n))
    used = sorted({x for f in faces for x in f})
    remap = {o: k for k, o in enumerate(used)}
    return Poly([new_v[k] for k in used], [[remap[x] for x in f] for f in faces])


def order_cycle(v, idx, n):
    pts = [v[i] for i in idx]
    g = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
    axis = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = crossf(n, axis)
    w = crossf(n, u)
    ang = [math.atan2(dotf(subf(p, g), w), dotf(subf(p, g), u)) for p in pts]
    return [idx[k] for k in sorted(range(len(idx)), key=lambda k: ang[k])]


def split(P, n, d, diag):
    """(P ∩ {n.x <= d}, P ∩ {n.x >= d}) with the split defect recorded."""
    lo = clip(P, n, d)
    hi = clip(P, (-n[0], -n[1], -n[2]), -d)
    v = P.volume()
    vl = lo.volume() if lo is not None else 0.0
    vh = hi.volume() if hi is not None else 0.0
    defect = abs(v - vl - vh)
    diag.max_defect = max(diag.max_defect, defect)
    diag.sum_defect += defect
    return lo, hi


def subtract_convex(pieces, Q_planes, diag):
    """(pieces inside Q, pieces outside Q) by BSP over Q's outward face planes."""
    inside, outside = [], []
    for P in pieces:
        cur = P
        for n, d in Q_planes:
            lo, hi = split(cur, n, d, diag)
            if hi is not None:
                outside.append(hi)
            cur = lo
            if cur is None:
                break
        if cur is not None:
            inside.append(cur)
    return inside, outside


# ----------------------------------------------------------------------------------------------- case from the manifest

class Case:
    def __init__(self, record: dict):
        self.name = record["name"]
        self.Lx, self.Ly, self.Z_top = record["Lx"], record["Ly"], record["Z_top"]
        self.h_exact = {}
        for key, txt in record["heights"].items():
            i, j = (int(s) for s in key.split(","))
            self.h_exact[(i, j)] = F(float.fromhex(txt)) if "/" not in txt else F(txt)
        self.diag = {tuple(int(s) for s in k.split(",")): v for k, v in record["diagonal"].items()}
        for (i, j), hv in self.h_exact.items():
            if F(float(hv)) != hv:
                raise GeometryRejected(f"node ({i}, {j}) is not exactly representable in binary64")

    def hx(self, i, j):
        return self.h_exact[(i % self.Lx, j % self.Ly)]

    def triangles(self, i, j):
        """(tag, exact vertices, float vertices) of the two triangles of column (i, j), unwrapped."""
        nd = {(a, b): self.hx(i + a, j + b) for a in (0, 1) for b in (0, 1)}
        P = lambda a, b: (F(i + a), F(j + b), nd[(a, b)])  # noqa: E731
        if self.diag[(i % self.Lx, j % self.Ly)] == "main":
            tris = [("T0", (P(0, 0), P(1, 0), P(1, 1))), ("T1", (P(0, 0), P(1, 1), P(0, 1)))]
        else:
            tris = [("T0", (P(0, 0), P(1, 0), P(0, 1))), ("T1", (P(1, 0), P(1, 1), P(0, 1)))]
        return [(tag, tri, tuple(tuple(float(c) for c in p) for p in tri)) for tag, tri in tris]


def exact_plane(tri):
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    a = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / det
    b = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det
    return a, b, z0 - a * x0 - b * y0


class Facet:
    def __init__(self, key, poly, n_raw_exact, a_float, unit, cell, plane_float, d_raw_exact):
        self.key, self.poly, self.n_raw_exact, self.a, self.unit, self.cell, self.plane = key, poly, n_raw_exact, a_float, unit, cell, plane_float
        self.d_raw_exact = d_raw_exact               # n_raw_exact . x = d_raw_exact on the facet (unwrapped position, exact)

    def sign(self, c):
        """Exact sign of c . a (= sign of c . n_raw, A_xy > 0)."""
        s = sum(F(c[k]) * self.n_raw_exact[k] for k in range(3))
        return (s > 0) - (s < 0)

    def centroid(self):
        """AREA centroid of the planar convex facet polygon (fan of triangles; #548: not the vertex mean)."""
        p0 = self.poly[0]
        wsum, acc = 0.0, [0.0, 0.0, 0.0]
        for q in range(1, len(self.poly) - 1):
            a, b = self.poly[q], self.poly[q + 1]
            cr = crossf(subf(a, p0), subf(b, p0))
            w = 0.5 * math.sqrt(dotf(cr, cr))
            wsum += w
            for k in range(3):
                acc[k] += w * (p0[k] + a[k] + b[k]) / 3.0
        return tuple(acc[k] / wsum for k in range(3))


def polygon_clip2(poly, n, d):
    out = []
    m = len(poly)
    for q in range(m):
        p, r = poly[q], poly[(q + 1) % m]
        fp, fr = n[0] * p[0] + n[1] * p[1] - d, n[0] * r[0] + n[1] * r[1] - d
        tol = EPS_ON * max(1.0, abs(d))
        cp = 0 if abs(fp) <= tol else (1 if fp > 0 else -1)
        cr = 0 if abs(fr) <= tol else (1 if fr > 0 else -1)
        if cp <= 0:
            out.append(p)
        if cp * cr < 0:
            s = fp / (fp - fr)
            out.append((p[0] + s * (r[0] - p[0]), p[1] + s * (r[1] - p[1])))
    if len(out) < 3:
        return None
    area = 0.5 * sum(out[q][0] * out[(q + 1) % len(out)][1] - out[(q + 1) % len(out)][0] * out[q][1] for q in range(len(out)))
    return out if abs(area) > 0.0 else None


class Builder:
    def __init__(self, record: dict):
        self.case = Case(record)
        self.diag = Diag()
        self.cells, self.facets, self._fcache = {}, {}, {}
        c = self.case
        zmin = math.floor(min(c.h_exact.values()))
        for i in range(c.Lx):
            for j in range(c.Ly):
                nodes = [c.hx(i + a, j + b) for a in (0, 1) for b in (0, 1)]
                for k in range(zmin, c.Z_top):
                    if k + 1 <= min(nodes):
                        continue                     # empty (exact classification)
                    pieces = self.cell_pieces(i, j, k)
                    vol, mom = 0.0, [0.0, 0.0, 0.0]
                    for pc in pieces:
                        v, m = pc.moment()
                        vol += v
                        mom = [mom[q] + m[q] for q in range(3)]
                    if not pieces:
                        raise GeometryRejected(f"cell ({i}, {j}, {k}) classified fluid but has no piece")
                    full = k >= max(nodes)
                    if full:
                        # #558: a FULL cell is the exact cube by the exact classification: analytic volume 1 and centre
                        # (i + 1/2, j + 1/2, k + 1/2) (task26-diag-sampler-1 class B: the FP64 fan centroid carried ulp errors);
                        # the cut-cell and facet algorithms are unchanged
                        self.cells[(i, j, k)] = {"volume": 1.0, "centroid": (i + 0.5, j + 0.5, k + 0.5), "pieces": pieces, "class": "full",
                                                 "fan_volume": vol}
                    else:
                        self.cells[(i, j, k)] = {"volume": vol, "centroid": tuple(mom[q] / vol for q in range(3)), "pieces": pieces,
                                                 "class": "cut"}
                    for f in self.terrain_facets(i, j, k):
                        self.facets[f.key] = f
                lid = self.lid_facet(i, j)
                self.facets[lid.key] = lid
        self.h, self.W, self.direct, self.vi = {}, {}, {}, {}

    def cell_pieces(self, i, j, k):
        out = []
        for tag, tri, trif in self.case.triangles(i, j):
            a, b, g = (float(x) for x in exact_plane(tri))
            P = Poly.box((float(i), float(j), float(k)), (float(i + 1), float(j + 1), float(k + 1)))
            ln = math.sqrt(a * a + b * b + 1.0)
            planes = [((a / ln, b / ln, -1.0 / ln), -g / ln)]
            foot = [(p[0], p[1]) for p in trif]
            area = 0.5 * ((foot[1][0] - foot[0][0]) * (foot[2][1] - foot[0][1]) - (foot[2][0] - foot[0][0]) * (foot[1][1] - foot[0][1]))
            if area < 0:
                foot = foot[::-1]
            for q in range(3):
                (x0, y0), (x1, y1) = foot[q], foot[(q + 1) % 3]
                nn = (y1 - y0, -(x1 - x0))
                l2 = math.hypot(*nn)
                planes.append(((nn[0] / l2, nn[1] / l2, 0.0), (nn[0] * x0 + nn[1] * y0) / l2))
            for n, d in planes:
                P = clip(P, n, d) if P is not None else None
            if P is not None:
                out.append(P)
        return out

    def terrain_facets(self, i, j, k):
        key = (i, j, k)
        if key in self._fcache:
            return self._fcache[key]
        out = []
        c = self.case
        for tag, tri, trif in c.triangles(i, j):
            ae, be, ge = exact_plane(tri)
            if ae == 0 and be == 0 and math.floor(ge) != k:
                continue
            a, b, g = float(ae), float(be), float(ge)
            foot = [(p[0], p[1]) for p in trif]
            area = 0.5 * ((foot[1][0] - foot[0][0]) * (foot[2][1] - foot[0][1]) - (foot[2][0] - foot[0][0]) * (foot[1][1] - foot[0][1]))
            if area < 0:
                foot = foot[::-1]
            part = polygon_clip2(foot, (a, b), k + 1 - g)
            part = polygon_clip2(part, (-a, -b), g - k) if part is not None else None
            if part is None:
                continue
            axy = 0.5 * sum(part[q][0] * part[(q + 1) % len(part)][1] - part[(q + 1) % len(part)][0] * part[q][1] for q in range(len(part)))
            n_raw = (-a, -b, 1.0)
            avec = (axy * n_raw[0], axy * n_raw[1], axy * n_raw[2])
            ln = math.sqrt(dotf(n_raw, n_raw))
            unit = (n_raw[0] / ln, n_raw[1] / ln, n_raw[2] / ln)
            poly = [(p[0], p[1], a * p[0] + b * p[1] + g) for p in part]
            out.append(Facet(("T", i % c.Lx, j % c.Ly, tag, k), poly, (-ae, -be, F(1)), avec, unit, (i % c.Lx, j % c.Ly, k),
                             (unit, g / ln), ge))
        self._fcache[key] = out
        return out

    def lid_facet(self, i, j):
        c = self.case
        Z = float(c.Z_top)
        poly = [(float(i), float(j), Z), (float(i + 1), float(j), Z), (float(i + 1), float(j + 1), Z), (float(i), float(j + 1), Z)]
        return Facet(("L", i % c.Lx, j % c.Ly), poly, (F(0), F(0), F(-1)), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0), (i % c.Lx, j % c.Ly, c.Z_top - 1),
                     ((0.0, 0.0, -1.0), -Z), F(-c.Z_top))

    def candidates(self, i, j, k, exclude=None):
        c = self.case
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                img = (ii - ii % c.Lx, jj - jj % c.Ly)
                for kk in (k - 1, k, k + 1):
                    for f in self.terrain_facets(ii, jj, kk):
                        if f.key != exclude:
                            out.append((f, img))
                if c.Z_top - 1 in (k - 1, k, k + 1):
                    lid = self.lid_facet(ii, jj)
                    if lid.key != exclude:
                        out.append((lid, img))
        return out

    def producer_pass(self, fkey, d):
        """Row (fkey, d): immediate region h per (cell, image) and W per first receiver (Cartesian BSP)."""
        f = self.facets[fkey]
        c = LEX19[d]
        if f.sign(c) <= 0:
            return
        vi = dotf(f.a, c)
        if not (math.isfinite(vi) and vi > 0.0):
            raise GeometryInvalid(f"exact-positive row {fkey} {c} has FP64 vi {vi}")     # #549
        self.vi[(fkey, d)] = vi
        i0 = math.floor(min(p[0] for p in f.poly))
        j0 = math.floor(min(p[1] for p in f.poly))
        recv = [(g, img) for g, img in self.candidates(i0, j0, f.cell[2], exclude=fkey) if g.sign(c) < 0]
        free, owned = self.producer_partition(f, c, recv)
        for p in free:
            self.add_h(fkey, d, p)
        for p, (g, img) in owned:
            self.add_w(fkey, d, g, img, p)

    def producer_partition(self, f, c, recv):
        """(free pieces, [(piece, (receiver, image))]) of the outgoing prism of producer f along c (#562).
        For x = y_b + t c in the prism, sigma_b(x) = t. A receiver a is ELIGIBLE at x only when
        0 <= sigma_a(x) <= sigma_b(x) <= 1: membership in a's forward prism gives 0 <= sigma_a <= 1 only; the
        after-producer half-space sigma_a <= sigma_b, i.e.
            (n_a/(n_a.c) - n_b/(n_b.c)) . x <= d_a/(n_a.c) - d_b/(n_b.c),
        gives the rest. Parallel planes have a constant order, decided exactly: a receiver whose plane is parallel and
        behind (or on) the producer's plane is never eligible. The first receiver is the eligible one with the LARGEST
        sigma_a (nearest the producer)."""
        O = Poly.prism(f.poly, c)
        nb, db = f.plane
        cb = dotf(nb, c)
        state = [(O, [])]                            # (piece, eligible receivers containing it)
        for g, img in recv:
            planes = Poly.prism(g.poly, c).halfspaces()      # g is already unwrapped at its image (#548)
            if exact_parallel(g.n_raw_exact, f.n_raw_exact):
                if exact_sigma_difference(g, f, c) >= 0:     # sigma_a >= sigma_b everywhere: never after the producer
                    continue
            else:
                na, da = g.plane
                ca = dotf(na, c)
                n = tuple(na[k] / ca - nb[k] / cb for k in range(3))
                ln = math.sqrt(dotf(n, n))
                planes = planes + [(tuple(x / ln for x in n), (da / ca - db / cb) / ln)]
            nxt = []
            for p, owners in state:
                pin, pout = subtract_convex([p], planes, self.diag)
                nxt.extend((q, owners + [(g, img)]) for q in pin)
                nxt.extend((q, owners) for q in pout)
            state = nxt
        free, owned = [], []
        for p, owners in state:
            if owners:
                self.first_owner(p, owners, c, owned)
            else:
                free.append(p)
        return free, owned

    def first_owner(self, p, owners, c, owned):
        """Splits p by order planes; the first receiver from the producer has the LARGEST back distance sigma.
        Appends (piece, (receiver, image)) to owned."""
        if len(owners) == 1:
            owned.append((p, owners[0]))
            return
        g, img = owners[0]
        planes = []
        na, da = g.plane                             # unwrapped plane (#548)
        for g2, img2 in owners[1:]:
            if exact_parallel(g.n_raw_exact, g2.n_raw_exact):
                # #549: parallel crossing planes have a CONSTANT order along every ray: decide it exactly; only equal
                # crossing times (the same plane) over a positive overlap are an R4 tie
                s = exact_sigma_difference(g, g2, c)
                if s == 0:
                    raise GeometryRejected(f"R4: coincident receiver planes {g.key} {g2.key} over an overlap")
                if s < 0:                            # g2 is nearer the producer on every ray: g is never first here
                    self.first_owner(p, owners[1:], c, owned)
                    return
                continue                             # g beats g2 everywhere: no plane needed
            nb, db = g2.plane
            # sigma_a >= sigma_b  <=>  (n_b/(n_b.c) - n_a/(n_a.c)).x <= d_b/(n_b.c) - d_a/(n_a.c)
            ca, cb = dotf(na, c), dotf(nb, c)
            n = tuple(nb[k] / cb - na[k] / ca for k in range(3))
            ln = math.sqrt(dotf(n, n))
            planes.append((tuple(x / ln for x in n), (db / cb - da / ca) / ln))
        win, lose = subtract_convex([p], [(n, dd) for n, dd in planes], self.diag) if planes else ([p], [])
        for q in win:
            owned.append((q, (g, img)))
        for q in lose:
            self.first_owner(q, owners[1:], c, owned)

    def add_w(self, fkey, d, g, img, p):
        v = checked_volume(p, ("W", fkey, d, g.key, img))
        self.note_small(v)
        key = (fkey, d, g.key, img)
        self.W[key] = self.W.get(key, 0.0) + v

    def add_h(self, fkey, d, p):
        xs = [q[0] for q in p.v]
        ys = [q[1] for q in p.v]
        zs = [q[2] for q in p.v]
        c = self.case
        for ci in range(math.floor(min(xs)), math.ceil(max(xs))):
            for cj in range(math.floor(min(ys)), math.ceil(max(ys))):
                for ck in range(math.floor(min(zs)), math.ceil(max(zs))):
                    q = p
                    for n, dd in Poly.box((float(ci), float(cj), float(ck)), (float(ci + 1), float(cj + 1), float(ck + 1))).halfspaces():
                        q = clip(q, n, dd) if q is not None else None
                    if q is None:
                        continue
                    v = checked_volume(q, ("h", fkey, d, ci, cj, ck))
                    self.note_small(v)
                    cell = (ci % c.Lx, cj % c.Ly, ck)
                    img = (ci - ci % c.Lx, cj - cj % c.Ly)
                    key = (fkey, d, cell, img)
                    self.h[key] = self.h.get(key, 0.0) + v

    def note_small(self, v):
        if v <= SMALL:
            self.diag.small_count += 1
            self.diag.small_total += v

    def donor_pass(self, cell, d):
        c = LEX19[d]
        i, j, k = cell
        pieces = list(self.cells[cell]["pieces"])
        for g, img in self.candidates(i, j, k):
            if g.sign(c) >= 0:
                continue
            S = Poly.prism(g.poly, (-c[0], -c[1], -c[2]))     # unwrapped at its image (#548)
            _, pieces = subtract_convex(pieces, S.halfspaces(), self.diag)
            if not pieces:
                break
        v = 0.0
        for p in pieces:
            pv = checked_volume(p, ("direct", cell, d))
            self.note_small(pv)
            v += pv
        self.direct[(cell, d)] = v

    def run(self, progress=None):
        for n, fkey in enumerate(sorted(self.facets, key=str)):
            for d in range(19):
                if LEX19[d] != (0, 0, 0):
                    before = self.diag.sum_defect
                    self.producer_pass(fkey, d)
                    self.diag.max_unit_sum = max(self.diag.max_unit_sum, self.diag.sum_defect - before)
            if progress:
                progress("facet", n)
        for n, cell in enumerate(sorted(self.cells)):
            for d in range(19):                      # all 19, including REST (direct = V into itself, as task 22, #541)
                before = self.diag.sum_defect
                self.donor_pass(cell, d)
                self.diag.max_unit_sum = max(self.diag.max_unit_sum, self.diag.sum_defect - before)
            if progress:
                progress("cell", n)


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="ascii"))


# ----------------------------------------------------------------------------------------------- sampler and visibility

VIS_MARGIN = 1e-12


def surface_height(case: Case, x: float, y: float) -> float:
    i, j = math.floor(x), math.floor(y)
    for tag, tri, trif in case.triangles(i, j):
        foot = [(p[0], p[1]) for p in trif]
        if point_in_triangle(foot, (x, y)):
            a, b, g = (float(v) for v in exact_plane(tri))
            return a * x + b * y + g
    raise GeometryRejected(f"no triangle contains ({x}, {y})")


def point_in_triangle(t, p):
    def s(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3 = s(t[0], t[1], p), s(t[1], t[2], p), s(t[2], t[0], p)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def segment_free(case: Case, p, q) -> bool:
    """True when z - h_T >= VIS_MARGIN along the segment p -> q (piecewise linear: knots at x, y integer lines and the
    column diagonals), and both ends lie below the lid."""
    ts = {0.0, 1.0}
    dx, dy = q[0] - p[0], q[1] - p[1]
    lo_x, hi_x = sorted((p[0], q[0]))
    lo_y, hi_y = sorted((p[1], q[1]))
    for n in range(math.floor(lo_x), math.ceil(hi_x) + 1):
        if dx != 0:
            ts.add((n - p[0]) / dx)
    for m in range(math.floor(lo_y), math.ceil(hi_y) + 1):
        if dy != 0:
            ts.add((m - p[1]) / dy)
    for s in range(math.floor(lo_x + lo_y) - 1, math.ceil(hi_x + hi_y) + 2):
        if dx + dy != 0:
            ts.add((s - p[0] - p[1]) / (dx + dy))
    for s in range(math.floor(lo_x - hi_y) - 1, math.ceil(hi_x - lo_y) + 2):
        if dx - dy != 0:
            ts.add((s - p[0] + p[1]) / (dx - dy))
    for t in sorted(x for x in ts if 0.0 <= x <= 1.0):
        x, y, z = (p[k] + t * (q[k] - p[k]) for k in range(3))
        if z - surface_height(case, x, y) < VIS_MARGIN or z > case.Z_top:
            return False
    return True


def build_sampler(b: Builder, stencil_weights) -> dict:
    """x_s = facet centroid + n/2; 12 nearest 3D cell centroids with x/y images (no z image), the accepted task-22
    stencil_weights (direct_centroid / least squares with its gates); R2 and R3 visibility by segment_free."""
    c = b.case
    keys = sorted(b.cells)
    cents = [b.cells[k]["centroid"] for k in keys]
    rows, cols, vals, branches, conds, repro = [], [], [], {"direct_centroid": 0, "least_squares": 0}, [], []
    facet_keys = sorted(b.facets, key=str)
    points = []
    per_facet_branch = {}
    for fi, fkey in enumerate(facet_keys):
        f = b.facets[fkey]
        cen = f.centroid()
        xs = tuple(cen[k] + 0.5 * f.unit[k] for k in range(3))
        if not segment_free(c, cen if fkey[0] == "L" else tuple(cen[k] + 1e-9 * f.unit[k] for k in range(3)), xs):
            raise GeometryRejected(f"R2: sample of facet {fkey} not visible")
        cand = []
        for sx in (-c.Lx, 0, c.Lx):
            for sy in (-c.Ly, 0, c.Ly):
                for idx, ce in enumerate(cents):
                    delta = np.array([ce[0] + sx - xs[0], ce[1] + sy - xs[1], ce[2] - xs[2]])
                    cand.append((float(delta @ delta), idx, 0, sx, sy, delta))
        cand.sort(key=lambda z: (z[0], z[1], z[3], z[4]))
        try:
            branch, chosen, coeff, cond, rep = stencil_weights(cand, 1)
        except Exception as e:  # noqa: BLE001  #558: the rejection carries the facet, sample and nearest distance
            raise GeometryRejected(f"{e} at facet {fkey} sample {[float(v).hex() for v in xs]} nearest d2 {float(cand[0][0]).hex()}") from e
        for z in chosen:
            ce = cents[z[1]]
            if not segment_free(c, xs, (ce[0] + z[3], ce[1] + z[4], ce[2])):
                raise GeometryRejected(f"R3: stencil cell {keys[z[1]]} not visible from the sample of facet {fkey}")
        branches[branch] += 1
        per_facet_branch[str(fkey)] = branch
        if branch == "least_squares":
            conds.append(cond)
            repro.append(rep)
        for z, v in zip(chosen, coeff):
            rows.append(fi); cols.append(z[1]); vals.append(float(v))
        points.append(xs)
    if repro and (max(repro) > 1e-10 or max(conds) > 1e12):
        raise GeometryRejected(f"R3: reproduction {max(repro)} / condition {max(conds)}")
    return {"keys": keys, "facet_keys": facet_keys, "rows": rows, "cols": cols, "vals": vals, "points": points, "branches": branches,
            "per_facet_branch": per_facet_branch,
            "reproduction_max": max(repro) if repro else None, "condition_max": max(conds) if conds else None}
