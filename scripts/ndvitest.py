import rasterio

with rasterio.open('Wegrzce_tify/01W1509.tif') as src:
    # 1. Opisy poszczególnych bandów (często to tutaj oprogramowanie drona zapisuje długość fali w nm)
    print("Opisy bandów:")
    for i, desc in enumerate(src.descriptions, start=1):
        print(f"Band {i}: {desc}")
        
    print("\n" + "-"*40 + "\n")

    # 2. Globalne metadane całego pliku (często zawiera informacje o sensorze)
    print("Globalne tagi pliku:")
    for key, value in src.tags().items():
        print(f"{key}: {value}")

    print("\n" + "-"*40 + "\n")

    # 3. Tagi przypisane bezpośrednio do konkretnego bandu (np. Band 1)
    # Sprawdzamy dla pierwszych 3 bandów, żeby nie zalać konsoli tekstem
    for i in range(1, 4):
        print(f"Tagi dla Bandu {i}:", src.tags(i))