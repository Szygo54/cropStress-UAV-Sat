import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. KONFIGURACJA ŚCIEŻKI ---
drone_path = 'Wegrzce_tify/01W1509.tif'  # Podmień na właściwą nazwę pliku z drona

# --- 2. WCZYTANIE I RZUTOWANIE DANYCH Z DRONA ---
print("Wczytywanie rastra z kamery FigSpec...")
drone = rioxarray.open_rasterio(drone_path)

print("Rzutowanie do układu metrycznego PUWG 1992 (EPSG:2180)...")
# Ta linijka naprawia osie wykresu i przygotowuje plik pod analizy przestrzenne w metrach
drone = drone.rio.reproject("EPSG:2180")

# --- 3. WYBÓR PASM (Zgodnie z metadanymi) ---
print("Wyodrębnianie pasm Red i NIR...")
red_drone = drone.isel(band=15).astype(float)
nir_drone = drone.isel(band=21).astype(float)

# --- 4. OBLICZANIE WSKAŹNIKA NDVI ---
print("Obliczanie wskaźnika NDVI...")
np.seterr(divide='ignore', invalid='ignore')
drone_ndvi = (nir_drone - red_drone) / (nir_drone + red_drone)

# Ręczne odcięcie gleby
NDVI_THRESHOLD = 0.06
print(f"Maskowanie gleby (pozostawiono NDVI > {NDVI_THRESHOLD})...")
drone_ndvi_masked = drone_ndvi.where(drone_ndvi > NDVI_THRESHOLD)

# --- 5. WIZUALIZACJA ---
print("Generowanie podglądu...")
plt.figure(figsize=(10, 8))

# Pobranie poprawnych, metrycznych granic przestrzennych
minx, miny, maxx, maxy = drone.rio.bounds()

# Rysowanie mapy
im = plt.imshow(drone_ndvi_masked.squeeze(), cmap='RdYlGn', vmin=0.0, vmax=1.0, 
                extent=[minx, maxx, miny, maxy], origin='upper')

plt.colorbar(im, label='Wartość NDVI')
plt.title('Wskaźnik NDVI z drona FigSpec (Tylko roślinność)', fontweight='bold')
plt.xlabel('Współrzędna X (m)')
plt.ylabel('Współrzędna Y (m)')

# Obrót etykiet osi X o 45 stopni dla lepszej czytelności długich liczb
plt.xticks(rotation=45)
plt.tight_layout()

# Wyświetlenie okna z wykresem bez zapisywania pliku
plt.show()