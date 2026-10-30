import json
import rasterio
from rasterio.warp import transform_geom

tif_path = "Wegrzce_tify/01W1509.tif"  # Ścieżka do dowolnego nalotu z Węgrzc
geojson_path = "results/wegrzce_aoi.geojson"

with rasterio.open(tif_path) as src:
    b = src.bounds
    
    # 1. Budujemy prostokąt (Polygon) z narożników TIFF-a
    geom_orig = {
        "type": "Polygon",
        "coordinates": [[
            [b.left, b.bottom],
            [b.left, b.top],
            [b.right, b.top],
            [b.right, b.bottom],
            [b.left, b.bottom]  # Zamknięcie poligonu (pierwszy i ostatni punkt taki sam)
        ]]
    }
    
    # 2. Transformujemy z układu TIFF-a na WGS84 (EPSG:4326) wymagany przez Planet API
    geom_wgs84 = transform_geom(src.crs, "EPSG:4326", geom_orig)

# 3. Opakowujemy w pełną strukturę pliku GeoJSON (do podglądu np. w QGIS)
geojson_data = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"name": "Pole_Wegrzce"},
        "geometry": geom_wgs84
    }]
}

with open(geojson_path, "w", encoding="utf-8") as f:
    json.dump(geojson_data, f, indent=2)

print(f"Gotowe! Zapisano zasięg do pliku: {geojson_path}")