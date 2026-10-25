import os
import rioxarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. KONFIGURACJA ŚCIEŻEK ---
drone_path = '02.tif'
s2_b4_jp2 = 'B04_10m.jp2'  # Czerwień (Red) Sentinel
s2_b8_jp2 = 'B08_10m.jp2'  # Bliska podczerwień (NIR) Sentinel

s2_b4_tif = s2_b4_jp2.replace('.jp2', '.tif')
s2_b8_tif = s2_b8_jp2.replace('.jp2', '.tif')

# --- 2. AUTOMATYCZNA KONWERSJA DO GEOTIFF ---
for jp2_path, tif_path in zip([s2_b4_jp2, s2_b8_jp2], [s2_b4_tif, s2_b8_tif]):
    if not os.path.exists(tif_path):
        if os.path.exists(jp2_path):
            print(f"Konwersja {jp2_path} do formatu GeoTIFF...")
            data = rioxarray.open_rasterio(jp2_path)
            data.rio.to_raster(tif_path)
        else:
            raise FileNotFoundError(f"Krytyczny błąd: Brakuje pliku {jp2_path}!")

# --- 3. WCZYTANIE I RZUTOWANIE DANYCH ---
print("Wczytywanie rastrów...")
drone = rioxarray.open_rasterio(drone_path)
s2_b4 = rioxarray.open_rasterio(s2_b4_tif)
s2_b8 = rioxarray.open_rasterio(s2_b8_tif)

drone_rep = drone.rio.reproject(s2_b4.rio.crs)

# Przycięcie Sentinela do prostokątnego zasięgu drona
bounds = drone_rep.rio.bounds()
s2_b4_clipped = s2_b4.rio.clip_box(*bounds)
s2_b8_clipped = s2_b8.rio.clip_box(*bounds)

# --- 4. OBLICZANIE NDVI (DRON) I RĘCZNE MASKOWANIE ---
print("Obliczanie wskaźnika NDVI z drona...")
red_drone = drone_rep.isel(band=13).astype(float)
nir_drone = drone_rep.isel(band=22).astype(float)

np.seterr(divide='ignore', invalid='ignore')
drone_ndvi = (nir_drone - red_drone) / (nir_drone + red_drone)

# ---> TUTAJ USTALASZ SWÓJ SZTYWNY PRÓG <---
NDVI_THRESHOLD = 0.04  

print(f"Ręczne ustalanie progu odcięcia gleby: {NDVI_THRESHOLD}")

# Pobranie ważnych pikseli do wygenerowania histogramu na końcu skryptu
valid_ndvi = drone_ndvi.values[np.isfinite(drone_ndvi.values)]

print("Filtrowanie gleby...")
drone_ndvi_masked = drone_ndvi.where(drone_ndvi > NDVI_THRESHOLD)
drone_ndvi_masked.rio.write_nodata(np.nan, inplace=True)
drone_ndvi_masked.rio.write_crs(drone_rep.rio.crs, inplace=True)

# --- 5. UPSCALING I PRZETWARZANIE SENTINELA ---
print("Upscaling biomasy do rozdzielczości 10x10 m (Resampling.average)...")
drone_ndvi_10m = drone_ndvi_masked.rio.reproject_match(s2_b4_clipped, resampling=Resampling.average)

print("Obliczanie referencyjnego NDVI z Sentinela-2...")
red_s2 = s2_b4_clipped.squeeze().astype(float)
nir_s2 = s2_b8_clipped.squeeze().astype(float)
s2_ndvi = (nir_s2 - red_s2) / (nir_s2 + red_s2)

print("Przycinanie Sentinela do przestrzennej maski drona...")
s2_ndvi_masked = s2_ndvi.where(drone_ndvi_10m.notnull())

# --- 6. INTEGRACJA DANYCH (TABELA ML) ---
print("Składanie danych w tabelę ML...")
df_drone = drone_ndvi_10m.squeeze().to_dataframe(name='NDVI_Dron_Czyste_Liscie').reset_index()
df_s2 = s2_ndvi_masked.squeeze().to_dataframe(name='NDVI_Sentinel').reset_index()

df_combined = pd.merge(df_drone[['x', 'y', 'NDVI_Dron_Czyste_Liscie']], df_s2[['x', 'y', 'NDVI_Sentinel']], on=['x', 'y'])
df_clean = df_combined.dropna().reset_index(drop=True)

print(f"\n[SUKCES] Wygenerowano {len(df_clean)} czystych punktów treningowych.")
print(df_clean.head())

print("\nGenerowanie planszy (2x2)...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()  


vmin_val = np.nanpercentile(df_clean['NDVI_Dron_Czyste_Liscie'], 2)
vmax_val = np.nanpercentile(df_clean['NDVI_Dron_Czyste_Liscie'], 98)

print("\nGenerowanie planszy (2x2)...")
# Zmniejszono nieco rozmiar okna dla lepszych proporcji
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten() 

# Dodanie czytelnego tytułu głównego
fig.suptitle('Porównanie przestrzenne wskaźnika NDVI: Dron vs Sentinel-2', fontsize=16, fontweight='bold', y=0.96)

vmin_val = np.nanpercentile(df_clean['NDVI_Dron_Czyste_Liscie'], 2)
vmax_val = np.nanpercentile(df_clean['NDVI_Dron_Czyste_Liscie'], 98)

# -- OKNO 1: Dron FigSpec --
minx, miny, maxx, maxy = drone_rep.rio.bounds()
im = axes[0].imshow(drone_ndvi_masked.values, cmap='RdYlGn', vmin=vmin_val, vmax=vmax_val, 
               extent=[minx, maxx, miny, maxy], origin='upper')
axes[0].set_title('1. Dron FigSpec', fontweight='bold')

# -- OKNO 2: Dron uśredniony --
minx_s, miny_s, maxx_s, maxy_s = drone_ndvi_10m.rio.bounds()
axes[1].imshow(drone_ndvi_10m.values, cmap='RdYlGn', vmin=vmin_val, vmax=vmax_val, 
               extent=[minx_s, maxx_s, miny_s, maxy_s], origin='upper')
axes[1].set_title('2. Dron uśredniony (10x10 m)', fontweight='bold')

# -- OKNO 3: Sentinel-2 --
axes[2].imshow(s2_ndvi_masked.values, cmap='RdYlGn', vmin=vmin_val, vmax=vmax_val, 
               extent=[minx_s, maxx_s, miny_s, maxy_s], origin='upper')
axes[2].set_title('3. Sentinel-2', fontweight='bold')

# Estetyka osi dla map (okna 0, 1, 2)
for i in range(3):
    axes[i].set_xlabel('Współrzędna X (m)')
    axes[i].set_ylabel('Współrzędna Y (m)')
    # Obrót etykiet osi X o 45 stopni, aby długie liczby nie nachodziły na siebie
    axes[i].tick_params(axis='x', rotation=45)
    # Dodanie delikatnej siatki ułatwiającej orientację w terenie
    axes[i].grid(True, linestyle=':', alpha=0.6)

# Wyłączenie widoczności osi w czwartym oknie
axes[3].axis('off')

# Umieszczenie paska skali (colorbar) wewnątrz czwartego okna
# Dzięki 'location=left' pasek przysunie się estetycznie do środka planszy
cbar = fig.colorbar(im, ax=axes[3], location='left', fraction=0.3, aspect=12)
cbar.set_label('Wartość NDVI', fontsize=12, fontweight='bold')

# Horyzontalne i wertykalne dopasowanie marginesów
plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace=0.2, hspace=0.3) 
plt.show()