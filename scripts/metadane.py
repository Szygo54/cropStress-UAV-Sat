import rasterio

image_path = 'zdj1.tif'

with rasterio.open(image_path) as src:
    print(f"Liczba wszystkich pasm: {src.count}")
    
    # Odczytanie nazw/opisów poszczególnych pasm
    print("\nOpisy pasm:")
    for i, desc in enumerate(src.descriptions, start=1):
        print(f"Kanał {i}: {desc}")
        
    # Odczytanie ogólnych metadanych pliku (czasami długości fal są ukryte tutaj)
    print("\nPozostałe metadane TIFF:")
    for key, value in src.tags().items():
        print(f"{key}: {value}")