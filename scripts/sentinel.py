import os
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

# Ścieżki do plików
drone_path = '02.tif'
sentinel_jp2_path = 'B04_10m.jp2'
sentinel_tif_path = sentinel_jp2_path.replace('.jp2', '.tif')

# --- AUTOMATYZACJA KONWERSJI W CZYSTYM RIOXARRAY ---
# Skrypt sprawdza, czy przekonwertowany plik TIFF już istnieje. Jeśli nie - tworzy go.
if not os.path.exists(sentinel_tif_path):
    print(f"Wykryto plik {sentinel_jp2_path}. Trwa automatyczna konwersja do GeoTIFF...")
    jp2_data = rioxarray.open_rasterio(sentinel_jp2_path)
    jp2_data.rio.to_raster(sentinel_tif_path)
    print("Konwersja zakończona sukcesem!")

# 1. Wczytanie danych
print("Krok 1/5: Wczytywanie rastrów...")
drone = rioxarray.open_rasterio(drone_path)
sentinel = rioxarray.open_rasterio(sentinel_tif_path)

# 2. Wyrównanie układów współrzędnych
print("Krok 2/5: Rzutowanie drona do układu współrzędnych Sentinela (to może potrwać kilka minut)...")
drone_reprojected = drone.rio.reproject(sentinel.rio.crs)

# 3. Przycięcie Sentinela do fizycznych granic nalotu drona
print("Krok 3/5: Przycinanie obrazu satelitarnego do granic uprawy...")
minx, miny, maxx, maxy = drone_reprojected.rio.bounds()
sentinel_clipped = sentinel.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

# 4. Wyliczenie NDVI z drona (jako ładne tło do wizualizacji)
print("Krok 4/5: Obliczanie macierzy NDVI z drona...")
# Indeksy w xarray liczymy od 0 (Red = kanał 14 -> indeks 13, NIR = kanał 23 -> indeks 22)
red_bg = drone_reprojected.values[13].astype(float)
nir_bg = drone_reprojected.values[22].astype(float)

np.seterr(divide='ignore', invalid='ignore')
drone_ndvi = (nir_bg - red_bg) / (nir_bg + red_bg)

# 5. Odczytanie rzeczywistej siatki Sentinela
res_x, res_y = sentinel_clipped.rio.resolution()

edges_x = sentinel_clipped.x.values - (res_x / 2)
edges_x = np.append(edges_x, edges_x[-1] + res_x)

edges_y = sentinel_clipped.y.values - (res_y / 2)
edges_y = np.append(edges_y, edges_y[-1] + res_y)

# --- WIZUALIZACJA ---
print("Krok 5/5: Generowanie wykresu...")
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title('Siatka Sentinel-2 (10x10 m) nałożona na nalot FigSpec', fontsize=14)

vmin_val = np.nanpercentile(drone_ndvi, 2)
vmax_val = np.nanpercentile(drone_ndvi, 98)
im = ax.imshow(drone_ndvi, cmap='RdYlGn', vmin=vmin_val, vmax=vmax_val, 
               extent=[minx, maxx, miny, maxy], origin='upper')

# Rysowanie czerwonej siatki Sentinela z paskami postępu
for ex in tqdm(edges_x, desc="Rysowanie pionowych linii siatki", unit="linia"):
    ax.axvline(ex, color='red', linestyle='-', linewidth=1.5, alpha=0.8)

for ey in tqdm(edges_y, desc="Rysowanie poziomych linii siatki", unit="linia"):
    ax.axhline(ey, color='red', linestyle='-', linewidth=1.5, alpha=0.8)

ax.set_xlabel('Współrzędna X (metry)')
ax.set_ylabel('Współrzędna Y (metry)')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='NDVI z drona')
plt.tight_layout()
plt.show()