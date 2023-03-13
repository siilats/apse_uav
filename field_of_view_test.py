import numpy as np

h, w = 2160, 3840
h2, w2 = 1154.6619175555534*2, 2101.3043063061195*2
fx, fy = 3403.052978515625, 3434.074462890625



fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

print("Field of View (degrees):")
print(f"  {fov_x = :.1f}\N{DEGREE SIGN}")
print(f"  {fov_y = :.1f}\N{DEGREE SIGN}")