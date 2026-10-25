import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURACJA I WCZYTANIE DANYCH FIGSPEC
# ==========================================
uav_path = 'Wegrzce_tify/01W0310.tif'

# Zaktualizowany słownik pasm (bez green_550, które było tylko do ARI)
UAV_BANDS = {
    'blue_450': 3,   # 442.7 nm
    'green_531': 7,  # 522.1 nm
    'green_570': 9,  # 561.5 nm
    'red_680': 15,   # 682.4 nm
    're_700': 16,    # 702.7 nm
    're_705': 16,    # 702.7 nm
    'nir_750': 18,   # 741.2 nm
    'nir_800': 21    # 801.8 nm
} 

d = {}
with rasterio.open(uav_path) as src:
    for name, b_idx in UAV_BANDS.items():
        band = src.read(b_idx).astype('float32')
        
        # 1. Maskowanie oficjalnych wartości NoData pliku
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
            
        # 2. Maskowanie czarnych ramek / tła drona (wartości 0) na NaN
        band[band == 0] = np.nan
        
        d[name] = band

# ==========================================
# 2. OBLICZANIE WSKAŹNIKÓW (6 sztuk, bez ARI)
# ==========================================
eps = 1e-8
indices = {}

indices['NDVI'] = (d['nir_800'] - d['red_680']) / (d['nir_800'] + d['red_680'] + eps)

indices['EVI'] = 2.5 * ((d['nir_800'] - d['red_680']) / 
                        (d['nir_800'] + 6 * d['red_680'] - 7.5 * d['blue_450'] + 1))

L = 0.50
indices['SAVI'] = ((d['nir_800'] - d['red_680']) / 
                   (d['nir_800'] + d['red_680'] + L + eps)) * (1 + L)

indices['RENDVI'] = (d['nir_750'] - d['re_705']) / (d['nir_750'] + d['re_705'] + eps)

indices['NDWI'] = (d['green_570'] - d['nir_800']) / (d['green_570'] + d['nir_800'] + eps)

indices['CIre'] = (d['nir_800'] / (d['re_705'] + eps)) - 1.0

# ==========================================
# 3. WIZUALIZACJA 6 WSKAŹNIKÓW
# ==========================================
# Zmieniona siatka na 2x3 (brak pustego miejsca)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

plots_config = [
    ('NDVI', 'RdYlGn', 0.1, 0.9),
    ('EVI', 'RdYlGn', 0.0, 0.8),
    ('SAVI', 'RdYlGn', 0.1, 0.7),
    ('RENDVI', 'RdYlGn', 0.0, 0.6),
    ('NDWI', 'BrBG', -0.5, 0.5),
    ('CIre', 'viridis', 0.0, 1.0) 
]

for i, (name, cmap_name, vmin, vmax) in enumerate(plots_config):
    # Skopiowanie palety kolorów, aby wymusić biały kolor dla tła (NaN)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='white')
    
    im = axes[i].imshow(indices[name], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[i].set_title(name, fontsize=14, fontweight='bold')
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()

fig.suptitle('Analiza wskaźników wegetacyjnych buraka cukrowego (FigSpec) 03.10', fontsize=18, fontweight='bold', y=1.05)
plt.savefig('wskazniki_z_drona_0310.png', dpi=300, bbox_inches='tight')
plt.show()