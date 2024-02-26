import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import utm

# filename = "Geophilus_aufbereitet_2023-04-05_09-51-26.txt"
# filename = "Geophilus_aufbereitet_2023-04-05_14-30-36.txt"
filename = "Geophilus_aufbereitet_2023-04-06_09-00-00.txt"
data = np.genfromtxt(filename, names=True, delimiter="\t")
print(data.dtype.names)
data["Rho5"] *= 60
# %%
x, y, zone, _ = utm.from_latlon(data["N"], data["E"])
dt = np.sqrt(np.diff(x)**2+np.diff(y)**2)
print(np.median(dt))
fig = plt.figure()
skiptoks = ["Datum", "Zeit", "GammaOK", "PSOK", "GPSOK", "N", "E", "Kurs",
            "Streckeabsolut", "VTGBearing"]
pl = dict(s=0.5, cmap="Spectral_r")
with PdfPages(filename.replace(".txt", ".pdf")) as pdf:
    for tok in data.dtype.names:
        if not tok.startswith(("GGA", "Log")) and tok not in skiptoks:
            fig.clf()
            ax = fig.add_subplot()
            vmin, vmax = None, None
            if tok.startswith("Rho"):
                vmin, vmax = 10, 3000
                norm = LogNorm(vmin, vmax)
                im = ax.scatter(x, y, c=data[tok], norm=norm, **pl)
            else:
                if tok.startswith("LogRho"):
                    vmin, vmax = 1, 3.5

                im = ax.scatter(x, y, c=data[tok], vmin=vmin, vmax=vmax, **pl)

            ax.set_aspect(1.0)
            plt.colorbar(im, ax=ax, orientation="vertical")
            ax.set_title(tok)
            fig.savefig(pdf, format='pdf')
# %%
phase = np.arccos(data["RhoPhase1"] / data["Rho1"]) * 180 / np.pi
fig, ax = plt.subplots()
im = ax.scatter(x, y, c=phase, s=.5, cmap="Spectral_r", vmin=-10, vmax=30)
plt.colorbar(im, ax=ax, orientation="vertical")
