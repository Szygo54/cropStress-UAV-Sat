import os
import rioxarray
import rasterio
from rasterio.merge import merge
from arosics import COREG
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

pliki_oryginalne = ['male1.tif', 'male2.tif', 'male3.tif']
pliki_metryczne = ['czesc1_2180.tif', 'czesc2_2180.tif', 'czesc3_2180.tif']
pliki_naprawione = ['czesc2_naprawiona.tif', 'czesc3_naprawiona.tif']

print("KROK 1: Zmiana układu współrzędnych ze stopni na metry (EPSG:2180)...")
for wejscie, wyjscie in zip(pliki_oryginalne, pliki_metryczne):
    if os.path.exists(wejscie):
        raster = rioxarray.open_rasterio(wejscie)
        raster_metryczny = raster.rio.reproject("EPSG:2180")
        raster_metryczny.rio.to_raster(wyjscie)
        print(f" -> Przekonwertowano {wejscie} na metry.")
    else:
        print(f"Błąd: Nie znaleziono pliku {wejscie}")

print("\nKROK 2: Dopasowywanie Części 2 do Części 1 (na metrach)...")
# Używamy plików z końcówką _2180
coreg_1_2 = COREG(pliki_metryczne[0], pliki_metryczne[1], 
                  path_out=pliki_naprawione[0], fmt_out='GTIFF',
                  r_b4match=13, s_b4match=13, 
                  ws=(256, 256), max_shift=150, nodata=(0, 0))
coreg_1_2.calculate_spatial_shifts()

if coreg_1_2.success:
    coreg_1_2.correct_shifts()
    print(f"Sukces Cześć 2! Przesunięcie X: {coreg_1_2.x_shift_px}px, Y: {coreg_1_2.y_shift_px}px")
else:
    print("Nadal brak dopasowania dla Części 2.")

print("\nKROK 3: Dopasowywanie Części 3 do naprawionej Części 2...")
# Referencją jest naprawiona część 2
if os.path.exists(pliki_naprawione[0]):
    coreg_2_3 = COREG(pliki_naprawione[0], pliki_metryczne[2], 
                      path_out=pliki_naprawione[1], fmt_out='GTIFF',
                      r_b4match=13, s_b4match=13, 
                      ws=(256, 256), max_shift=150, nodata=(0, 0))
    coreg_2_3.calculate_spatial_shifts()
    if coreg_2_3.success:
        coreg_2_3.correct_shifts()
        print(f"Sukces Cześć 3! Przesunięcie X: {coreg_2_3.x_shift_px}px, Y: {coreg_2_3.y_shift_px}px")
    else:
        print("Nadal brak dopasowania dla Części 3.")

print("\nKROK 4: Sklejanie naprawionych i rzutowanych plików...")
pliki_do_sklejenia = [pliki_metryczne[0], pliki_naprawione[0], pliki_naprawione[1]]
otwarte_rastry = [rasterio.open(fp) for fp in pliki_do_sklejenia if os.path.exists(fp)]

if len(otwarte_rastry) > 0:
    mosaic, out_trans = merge(otwarte_rastry, method='first')

    print("\nKROK 5: Generowanie podglądu...")
    band_preview = mosaic[12, :, :] 
    plt.figure(figsize=(10, 8))
    plt.imshow(band_preview, cmap='gray')
    plt.colorbar(label='Wartości pikseli (Pasmo 13)')
    plt.title('Mozaika (EPSG:2180)', fontweight='bold')
    plt.tight_layout()
    plt.show()

    for src in otwarte_rastry:
        src.close()