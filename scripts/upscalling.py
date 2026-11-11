import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import transform_bounds
from shapely.geometry import box
from rasterio.features import geometry_mask
import gc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie obliczeniowe: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"Karta graficzna: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# 1. KONFIGURACJA KANAŁÓW
# ==============================================================================

UAV_BANDS = {
    'B450': 3,   # 442.7 nm (Blue)
    'B531': 7,   # 522.1 nm (Green 1)
    'B550': 8,   # 542.9 nm (Green 2)
    'B570': 9,   # 561.5 nm (Green 3)
    'B680': 15,  # 682.4 nm (Red)
    'B705': 16,  # 702.7 nm (Red-Edge)
    'B750': 18,  # 741.2 nm (NIR 1)
    'B800': 21   # 801.8 nm (NIR 2)
}

PS_BANDS = {
    'B2': 2,     # Blue (490 nm)
    'B3': 3,     # Green I (531 nm)
    'B4': 4,     # Green (565 nm)
    'B6': 6,     # Red (665 nm)
    'B7': 7,     # Red-Edge (705 nm)
    'B8': 8      # NIR (865 nm)
}

EPS = 1e-8

@torch.no_grad()
def compute_6_indices_gpu(uav_cropped, ndvi_threshold=0.10):
    """
    Liczy 6 wskaźników z drona na karcie graficznej (bez PRI)
    ze sztywną normalizacją radiometryczną pasma NIR do mediany 0.35.
    Kolejność kanałów na wejściu: [B450, B550, B680, B705, B750, B800] (6 kanałów).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.from_numpy(uav_cropped.astype(np.float32)).to(device)
    
    # Maska ważnych pikseli (kanał 5 to B800 - NIR)
    valid_mask = (t[5] > 0) & torch.isfinite(t[5])
    
    # Sztywna normalizacja radiometryczna do mediany NIR = 0.35
    # (gwarantuje identyczną skalę SAVI i EVI we wszystkich terminach nalotów!)
    sample_vals = t[5][valid_mask]
    if sample_vals.numel() > 0:
        med_nir = torch.median(sample_vals[::10]).item()
        if med_nir > 0:
            t = t * (0.35 / med_nir)
            
    b450, b550, b680, b705, b750, b800 = t[0], t[1], t[2], t[3], t[4], t[5]
    eps = 1e-8
    L = 0.5
    
    ndvi   = (b800 - b680) / (b800 + b680 + eps)
    savi   = ((b800 - b680) / (b800 + b680 + L + eps)) * (1.0 + L)
    evi    = 2.5 * (b800 - b680) / (b800 + 6.0 * b680 - 7.5 * b450 + 1.0 + eps)
    rendvi = (b750 - b705) / (b750 + b705 + eps)
    cire   = (b750 / (b705 + eps)) - 1.0
    ndwi   = (b550 - b800) / (b550 + b800 + eps)
    
    veg_mask = valid_mask & (ndvi > ndvi_threshold)
    
    uav_ind = {
        'NDVI':   ndvi.cpu().numpy(),
        'SAVI':   savi.cpu().numpy(),
        'EVI':    evi.cpu().numpy(),
        'RENDVI': rendvi.cpu().numpy(),
        'CIre':   cire.cpu().numpy(),
        'NDWI':   ndwi.cpu().numpy()
    }
    
    valid_mask_np = valid_mask.cpu().numpy()
    veg_mask_np   = veg_mask.cpu().numpy()
    
    del t, b450, b550, b680, b705, b750, b800, ndvi, savi, evi, rendvi, cire, ndwi
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return uav_ind, valid_mask_np, veg_mask_np

def compute_ps_indices_cpu(ps_cropped):
    """Liczy 6 wskaźników (bez PRI) oraz wyciąga 5 surowych pasm dla dociętego rastra PlanetScope."""
    def norm(a):
        a = a.astype(np.float32)
        v = a[np.isfinite(a) & (a > 0)]
        # PlanetScope Surface Reflectance domyślnie jest skalowane przez 10000
        if v.size > 0 and np.nanpercentile(v, 95) > 2.0:
            return a / 10000.0
        return a

    # Wyciągnięcie odpowiednich pasm (indeksy 0-based, bez B3 potrzebnego wcześniej tylko do PRI)
    b2 = norm(ps_cropped[PS_BANDS['B2'] - 1]) # Blue
    b4 = norm(ps_cropped[PS_BANDS['B4'] - 1]) # Green
    b6 = norm(ps_cropped[PS_BANDS['B6'] - 1]) # Red
    b7 = norm(ps_cropped[PS_BANDS['B7'] - 1]) # Red-Edge
    b8 = norm(ps_cropped[PS_BANDS['B8'] - 1]) # NIR

    L = 0.5
    EPS = 1e-8
    
    # Zwracamy zarówno 6 wskaźników, jak i 5 surowych pasm odbicia (do wzmocnienia Random Forest!)
    ps_features = {
        'NDVI':    (b8 - b6) / (b8 + b6 + EPS),
        'SAVI':    ((b8 - b6) / (b8 + b6 + L + EPS)) * (1.0 + L),
        'EVI':     2.5 * (b8 - b6) / (b8 + 6.0 * b6 - 7.5 * b2 + 1.0 + EPS),
        'RENDVI':  (b8 - b7) / (b8 + b7 + EPS),
        'CIre':    (b8 / (b7 + EPS)) - 1.0,
        'NDWI':    (b4 - b8) / (b4 + b8 + EPS),
        # Surowe pasma satelitarne (informacja o bezwzględnej jasności/albedo piksela):
        'Blue':    b2,
        'Green':   b4,
        'Red':     b6,
        'RedEdge': b7,
        'NIR':     b8
    }
    
    return ps_features, b8


def process_pair_with_geojson(
    uav_tif_path,
    ps_tif_path,
    geojson_path,
    field_id="01W",
    flight_date="2026-10-01",
    ps_date="2026-10-02",
    ndvi_soil_threshold=0.10,
    fvc_min_threshold=0.05
):
    t0 = time.time()
    records = []

    # Wymuszenie zwolnienia pamięci RAM i VRAM przed startem nowej pary
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Wczytanie poligonu pola z pliku GeoJSON
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"Plik GeoJSON jest pusty: {geojson_path}")

    with rasterio.open(ps_tif_path) as src_ps, rasterio.open(uav_tif_path) as src_uav:
        ps_crs, uav_crs = src_ps.crs, src_uav.crs

        gdf_ps = gdf.to_crs(ps_crs)
        gdf_uav = gdf.to_crs(uav_crs)
        
        field_geom_ps = gdf_ps.union_all() if hasattr(gdf_ps, 'union_all') else gdf_ps.unary_union

        # 2. Docięcie (clip) sceny PlanetScope (mały plik, tu mask() działa ułamek sekundy)
        ps_cropped, ps_transform = mask(src_ps, gdf_ps.geometry, crop=True, filled=True, nodata=0)
        ps_features, ps_nir_valid = compute_ps_indices_cpu(ps_cropped)
        del ps_cropped

        # 3. Lekkie wczytanie 6 kanałów UAV przez okno (Window) zamiast ciężkiego rasterio.mask()!
        uav_band_list = [
            UAV_BANDS['B450'], UAV_BANDS['B550'], UAV_BANDS['B680'],
            UAV_BANDS['B705'], UAV_BANDS['B750'], UAV_BANDS['B800']
        ]
        print(f" -> Docinanie UAV do GeoJSON ({field_id} | {flight_date}) i liczenie 6 wskaźników na GPU...")
        
        # Wyznaczamy okno (bounding box) poligonu w pikselach drona
        minx, miny, maxx, maxy = gdf_uav.total_bounds
        win_uav = from_bounds(minx, miny, maxx, maxy, transform=src_uav.transform)
        win_uav = win_uav.round_offsets().round_lengths()
        
        # Przycięcie okna do fizycznych wymiarów rastra UAV
        r_off = max(0, int(win_uav.row_off))
        c_off = max(0, int(win_uav.col_off))
        h_win = min(src_uav.height - r_off, int(win_uav.height))
        w_win = min(src_uav.width - c_off, int(win_uav.width))
        win_uav = rasterio.windows.Window(c_off, r_off, w_win, h_win)
        
        uav_transform = window_transform(win_uav, src_uav.transform)
        
        # Czytamy czyste numpy array (bez narzutu MaskedArray!)
        uav_cropped = src_uav.read(uav_band_list, window=win_uav)
        
        # Nakładamy płaską maskę 2D poligonu GeoJSON (zużywa 6x mniej RAM-u niż mask()!)
        outside_poly_2d = geometry_mask(
            gdf_uav.geometry,
            out_shape=(h_win, w_win),
            transform=uav_transform,
            invert=False  # True poza poligonem
        )
        uav_cropped[:, outside_poly_2d] = 0
        del outside_poly_2d

        # Liczymy wskaźniki na GPU i od razu kasujemy surowy raster z RAM-u
        uav_ind, valid_uav_mask, veg_uav_mask = compute_6_indices_gpu(
            uav_cropped, ndvi_threshold=ndvi_soil_threshold
        )
        del uav_cropped
        gc.collect()

        uav_h, uav_w = valid_uav_mask.shape
        rows, cols = ps_nir_valid.shape

        # 4. Upscaling do oczek 3x3 m z dociętej siatki PlanetScope
        for r in range(rows):
            for c in range(cols):
                if ps_nir_valid[r, c] <= 0 or np.isnan(ps_nir_valid[r, c]):
                    continue

                x_min, y_max = ps_transform * (c, r)
                x_max, y_min = ps_transform * (c + 1, r + 1)

                pixel_box = box(min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max))
                if not field_geom_ps.intersects(pixel_box.centroid):
                    continue

                if ps_crs != uav_crs:
                    uav_xmin, uav_ymin, uav_xmax, uav_ymax = transform_bounds(
                        ps_crs, uav_crs, min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max)
                    )
                else:
                    uav_xmin, uav_ymin = min(x_min, x_max), min(y_min, y_max)
                    uav_xmax, uav_ymax = max(x_min, x_max), max(y_min, y_max)

                win = from_bounds(uav_xmin, uav_ymin, uav_xmax, uav_ymax, transform=uav_transform)
                r0 = max(0, int(round(win.row_off)))
                r1 = min(uav_h, int(round(win.row_off + win.height)))
                c0 = max(0, int(round(win.col_off)))
                c1 = min(uav_w, int(round(win.col_off + win.width)))

                if r1 <= r0 or c1 <= c0:
                    continue

                patch_valid = valid_uav_mask[r0:r1, c0:c1]
                total_valid = np.count_nonzero(patch_valid)
                expected_area = (r1 - r0) * (c1 - c0)

                if total_valid < 0.5 * expected_area or total_valid == 0:
                    continue

                patch_veg = veg_uav_mask[r0:r1, c0:c1]
                veg_count = np.count_nonzero(patch_veg)

                fvc = veg_count / float(total_valid)
                if fvc <= fvc_min_threshold:
                    continue

                row_dict = {
                    'field_id': field_id,
                    'flight_date': flight_date,
                    'ps_date': ps_date,
                    'row': r,
                    'col': c,
                    'x_center': (x_min + x_max) / 2.0,
                    'y_center': (y_min + y_max) / 2.0,
                    'FVC': round(fvc, 4),
                    'n_uav_veg_pixels': veg_count
                }

                for idx_name, idx_full_arr in uav_ind.items():
                    vals = idx_full_arr[r0:r1, c0:c1][patch_veg]
                    vals = vals[np.isfinite(vals)]
                    row_dict[f'UAV_{idx_name}'] = float(np.mean(vals)) if vals.size > 0 else np.nan

                for feat_name, ps_arr in ps_features.items():
                    row_dict[f'PS_{feat_name}'] = float(ps_arr[r, c])

                records.append(row_dict)

        # Jawne zwolnienie wielkich macierzy przed wyjściem z funkcji!
        del uav_ind, valid_uav_mask, veg_uav_mask, ps_features, ps_nir_valid
        gc.collect()

    print(f" -> Gotowe w {time.time() - t0:.2f} s | Wyciągnięto {len(records)} oczek 3x3 m.")
    return pd.DataFrame(records).dropna()

# ==============================================================================
# 4. WPISANIE WSZYSTKICH NALOTÓW I URUCHOMIENIE PĘTLI
# ==============================================================================

# Tutaj dopisujesz kolejne słowniki dla każdego nalotu i pola (01W oraz 02W)
# Pamiętaj, aby dodać klucz 'geojson_path' z właściwą ścieżką do każdego nalotu.
kampania = [
    # --- 1. NALOT: 10.08 ---
    {
       "field_id": "01W",
        "flight_date": "2026-08-10",
       "ps_date": "2026-08-09",
       "uav_tif": "Wegrzce_tify/01W1008.tif",
        "ps_tif":  "results/planet/planet_2026-08-09.tif",
        "geojson_path": "results/geojson/wegrzce_aoi-10-08.geojson" # <-- Dodano ścieżkę
     },
    # --- 2. NALOT: 02.09 ---
    {
        "field_id": "01W",
        "flight_date": "2026-09-02",
        "ps_date": "2026-09-02",
        "uav_tif": "Wegrzce_tify/01W0209.tif",
        "ps_tif":  "results/planet/planet_2026-09-02.tif",
        "geojson_path": "results/geojson/wegrzce_aoi-02-09.geojson" # <-- Dodano ścieżkę
    },
    # --- 3. NALOT: 09.09 ---
    {
        "field_id": "01W",
        "flight_date": "2026-09-09",
        "ps_date": "2026-09-09",
        "uav_tif": "Wegrzce_tify/01W0909.tif",
        "ps_tif":  "results/planet/planet_2026-09-09.tif",
        "geojson_path": "results/geojson/wegrzce_aoi-09-09.geojson" # <-- Dodano ścieżkę
    },
    # --- 4. NALOT: 15.09 ---   
    {
        "field_id": "01W",
        "flight_date": "2026-09-15",
        "ps_date": "2026-09-16",
        "uav_tif": "Wegrzce_tify/01W1509.tif",
        "ps_tif":  "results/planet/planet_2026-09-16.tif",
        "geojson_path": "results/geojson/wegrzce_aoi-15-09.geojson" # <-- Dodano ścieżkę
    },
    # --- 5. NALOT: 01.10 ---
    {
        "field_id": "01W",
        "flight_date": "2026-10-01",
        "ps_date": "2026-10-02",
        "uav_tif": "Wegrzce_tify/01W0110.tif",
        "ps_tif":  "results/planet/planet_2026-10-02.tif",
        "geojson_path": "results/geojson/wegrzce_aoi-01-10.geojson" # <-- Dodano ścieżkę
    },
    # --- 6. NALOT: 03.10 ---
    {
        "field_id": "01W",
        "flight_date": "2026-10-03",
        "ps_date": "2026-10-04",
        "uav_tif": "Wegrzce_tify/01W0310.tif",
        "ps_tif":  "results/planet/planet_2026-10-04.tif",
        "geojson_path": "results/geojson/wegrzce_aoi-03-10.geojson" # <-- Dodano ścieżkę
    },
    # ---7. NALOT: 06.10 ---
    {
        "field_id":"O1W",
        "flight_date": "2026-10-06",
        "ps_date" : "2026-10-06",
        "uav_tif": "Wegrzce_tify/01W0610.tif",
        "ps_tif": "results/planet/planet_2026-10-06.tif",
        "geojson_path" : "results/geojson/wegrzce_aoi-03-10.geojson"
    }
]

# Pętla przechodząca po wszystkich wpisanych wyżej nalotach:
all_dfs = []

for item in kampania:
    # Upewnij się, że sprawdzamy istnienie pliku GeoJSON w warunku
    if os.path.exists(item["uav_tif"]) and os.path.exists(item["ps_tif"]) and os.path.exists(item["geojson_path"]):
        print(f"\nPrzetwarzanie: Pole {item['field_id']} | Dron: {item['flight_date']} vs Satelita: {item['ps_date']}")
        # Zakładam, że Twoja funkcja to teraz `process_pair_with_geojson`, jak w poprzedniej poprawionej wersji
        df_pair = process_pair_with_geojson(
            uav_tif_path=item["uav_tif"],
            ps_tif_path=item["ps_tif"],
            geojson_path=item["geojson_path"], # Przekazujemy nowo dodaną ścieżkę
            field_id=item["field_id"],
            flight_date=item["flight_date"],
            ps_date=item["ps_date"],
            ndvi_soil_threshold=0.10,  # Maska gleby NDVI > 0.1
            fvc_min_threshold=0.05     # Odrzucenie FVC <= 5%
        )
        all_dfs.append(df_pair)
    else:
        print(f"Pominięto (brak pliku na dysku): {item['flight_date']} ({item['field_id']})")
        # Opcjonalnie: wydrukuj, którego pliku brakuje, żeby było łatwiej zdebugować
        if not os.path.exists(item["uav_tif"]): print(f" -> Brak UAV: {item['uav_tif']}")
        if not os.path.exists(item["ps_tif"]): print(f" -> Brak PlanetScope: {item['ps_tif']}")
        if not os.path.exists(item["geojson_path"]): print(f" -> Brak GeoJSON: {item['geojson_path']}")

# Złączenie wszystkich nalotów w jedną dużą tabelę (~22 tys. wierszy) i zapis do CSV:
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    output_csv = "baza_kalibracyjna_UAV_PlanetScope_FVC.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"\n==========================================")
    print(f"GOTOWE! Zapisano plik: {output_csv}")
    print(f"Łączna liczba oczek 3x3 m w bazie: {len(final_df)}")
    print(final_df.head())