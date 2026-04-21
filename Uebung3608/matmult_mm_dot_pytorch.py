# === Unterschiede ===
#
# 1. torch.dot(a, b)
#
# Führt das Skalarprodukt (dot product) zweier 1D-Tensoren (Vektoren) durch.
# Ergebnis: ein Skalar (0D-Tensor).
#
# Beide Eingaben müssen 1D-Tensoren derselben Länge sein.
# Beispiel:
# import torch
#
# a = torch.tensor([1.0, 2.0, 3.0])
# b = torch.tensor([4.0, 5.0, 6.0])
#
# result = torch.dot(a, b)
# print(result)  # Output: tensor(32.)
#
# Rechnung:
# 1 × 4 + 2 × 5 + 3 × 6 = 32
#
# 2. torch.mm(a, b)
#
# Führt eine Matrix-Multiplikation zwischen zwei 2D-Tensoren (Matrizen) durch.
# Beide Eingaben müssen 2D-Tensoren sein.
# Die Anzahl der Spalten von a muss gleich der Anzahl der Zeilen von b sein.
#
# Beispiel:
# a = torch.tensor([[1.0, 2.0],
#                   [3.0, 4.0]])
# b = torch.tensor([[5.0, 6.0],
#                   [7.0, 8.0]])
#
# result = torch.mm(a, b)
# print(result)
#
# Output:
#
# tensor([[19., 22.],
#         [43., 50.]])
#
#
# Rechnung:
#
# [1×5+2×7    1×6+2×8  =  [ 19    22
# 3×5+4×7    3×6+4×8]       43    50]

# 3. torch.matmul(a, b)
# Allgemeinere Matrix-Multiplikation, die mit 1D, 2D oder höherdimensionalen Tensoren funktioniert.
# Entsprechend den Regeln der Broadcasting-Logik.
# Besonderheit:
# Wenn a und b:
#
# 1D: Ergebnis ist ein Skalar (wie torch.dot)
#
# 1D × 2D oder 2D × 1D: ergibt Vektor
#
# 2D × 2D: klassische Matrix-Multiplikation (wie torch.mm)
#
# nD × nD: Batch-Matrix-Multiplikation
#
# Beispiel 1 – wie dot():
# a = torch.tensor([1.0, 2.0, 3.0])
# b = torch.tensor([4.0, 5.0, 6.0])
# print(torch.matmul(a, b))  # Output: tensor(32.)
#
#  Beispiel 2 – wie mm():
# a = torch.tensor([[1.0, 2.0],
#                   [3.0, 4.0]])
# b = torch.tensor([[5.0, 6.0],
#                   [7.0, 8.0]])
# print(torch.matmul(a, b))
#
# Beispiel 3 – Batch-Matrix-Multiplikation (3D-Tensoren):
# a = torch.randn(10, 2, 3)  # 10 Matrizen (2×3)
# b = torch.randn(10, 3, 4)  # 10 Matrizen (3×4)
#
# result = torch.matmul(a, b)  # Result: (10, 2, 4)
# print(result.shape)  # torch.Size([10, 2, 4])
#
# Zusammenfassung
# Funktion	Eingabegrößen	Operation	Rückgabewert
# torch.dot()	1D × 1D	Skalarprodukt	Skalar (0D-Tensor)
# torch.mm()	2D × 2D	Matrix-Multiplikation	2D-Tensor
# torch.matmul()	1D/2D/nD × 1D/2D/nD	Allgemeine Matrix-Multiplikation	nD-Tensor


import torch

mat1 = [[2.0,1.0],[1.0,1.0]]
mat2 = [[0.5,0.5],[0.5,0.5]]

tensor1 = torch.tensor(mat1)
tensor2 = torch.tensor(mat2)

print(tensor1)
print(tensor2)

print(torch.mm(tensor1,tensor2))

# Lineare Transformation:
# torch.addmm():
# Die Funktion torch.addmm(input, mat1, mat2, *, beta=1, alpha=1) führt folgende Berechnung aus:
#
# Ergebnis = 𝛽 * input + 𝛼 * (mat1 * mat2)
# eine lineare Kombination aus einer Matrix und einer Matrixmultiplikation.
#
# Beispiel:
# torch.addmm(0.5 * M1, M1, M2, beta=1.0, alpha=0.1)
# Oder schöner, ohne manuelles Skalieren von M1 vorher:
# torch.addmm(M1, M1, M2, beta=0.5, alpha=0.1)
#
# Beispiel in Code:
# import torch
#
# M1 = torch.tensor([[1.0, 2.0],
#                    [3.0, 4.0]])
#
# M2 = torch.tensor([[5.0, 6.0],
#                    [7.0, 8.0]])
#
# # Berechnung: 0.5 * M1 + 0.1 * (M1 @ M2)
# result = torch.addmm(M1, M1, M2, beta=0.5, alpha=0.1)
#
# print(result)
#
# Vergleich zur manuellen Version:
# manual = 0.5 * M1 + 0.1 * torch.matmul(M1, M2)
#
# Beide liefern exakt dasselbe Ergebnis.
#
# Vorteile von torch.addmm():
#
# Performanter (da in einer einzigen Low-Level-Operation ausgeführt)
#
# Lesbarer bei solchen linearen Kombinationen
#
# Unterstützt optionales Out-Tensor-Argument (für In-Place-Berechnung)

erg = torch.addmm(tensor1, tensor1, tensor2, beta=0.5, alpha=0.1)

print(erg)