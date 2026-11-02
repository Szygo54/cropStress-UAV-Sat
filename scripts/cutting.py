from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

# ============================ USTAWIENIA ============================
REFERENCJA = "Wegrzce_tify/01W1509.tif"
DO_PRZYCIECIA = [
    "Wegrzce_tify/02W1008.tif",
    "Wegrzce_tify/02W0209.tif",
]
FOLDER_WYNIKOW = "Wegrzce_tify/przyciete"

# Próg dla czarnego tła w pasmach optycznych (piksele <= PROG_TLA są uznawane za tło)
PROG_TLA = 0.0
# ====================================================================

PASEK = 512  # liczba wierszy przetwarzanych naraz (oszczędność RAM dla plików 30-pasmowych)


def pobierz_wzorzec_tla(dataset):
    """Pobiera wartość piksela z lewego górnego rogu (0, 0), gdzie zawsze jest puste tło."""
    return dataset.read(window=Window(0, 0, 1, 1))[:, 0, 0]


def czy_piksel_istnieje(dane, maska_gdal, nodata, piksel_tla, prog_tla):
    """
    Sprawdza piksel po pikselu, gdzie faktycznie są dane pola (True = piksel istnieje, False = tło/pusty).
    dane: tablica (liczba_pasm, wys, szer)
    """
    # 1. Maska wewnętrzna GDAL / kanał alfa
    istnieje = maska_gdal > 0

    # 2. Brak NaN i Inf we wszystkich pasmach (np. wszystkich 30 pasmach)
    istnieje &= np.all(np.isfinite(dane), axis=0)

    # 3. Różne od wartości NoData zapisanej w metadanych pliku
    if nodata is not None and np.isfinite(nodata):
        istnieje &= ~np.any(np.isclose(dane, nodata, rtol=1e-5, atol=1e-5), axis=0)

    # 4. Różne od wzorcowego pustego piksela z rogu (0, 0)
    istnieje &= ~np.all(
        np.isclose(dane, piksel_tla[:, None, None], rtol=1e-5, atol=1e-5, equal_nan=True),
        axis=0,
    )

    # 5. Odrzucenie ujemnych wartości NoData (np. -9999, -3.4e38)
    istnieje &= np.all(dane > -100, axis=0)

    # 6. Odrzucenie czarnego tła (zera w pierwszych 3 pasmach lub we wszystkich pasmach)
    n_opt = min(3, dane.shape[0])
    istnieje &= np.any(dane[:n_opt] > prog_tla, axis=0)
    istnieje &= np.all(dane[:n_opt] != 0, axis=0)

    return istnieje


def dopasuj_i_przytnij(sciezka_src, ref, piksel_tla_ref, sciezka_wyj):
    with rasterio.open(sciezka_src) as src:
        piksel_tla_src = pobierz_wzorzec_tla(src)

        # Ustawiamy profil wyjściowy identyczny z siatką 01W1509.tif
        prof = ref.profile.copy()
        nodata_wyj = ref.nodata
        if nodata_wyj is None and np.issubdtype(src.dtypes[0], np.floating):
            if np.any(np.isnan(piksel_tla_ref)):
                nodata_wyj = np.nan

        prof.update(
             count=src.count,
             dtype=src.dtypes[0],
             nodata=nodata_wyj,
             compress="lzw",
             tiled=True,
             blockxsize=256,
             blockysize=256,
             BIGTIFF="IF_SAFER",
        )

        # Wartość wstawiana w miejsce skasowanych pikseli (taka sama jak tło w 01W1509.tif)
        if src.count == ref.count:
            wypelniacz_tla = piksel_tla_ref.astype(src.dtypes[0])
        elif np.issubdtype(src.dtypes[0], np.floating) and (nodata_wyj is None or np.isnan(nodata_wyj)):
            wypelniacz_tla = np.full(src.count, np.nan, dtype=src.dtypes[0])
        else:
            wart = 0 if nodata_wyj is None else nodata_wyj
            wypelniacz_tla = np.full(src.count, wart, dtype=src.dtypes[0])

        zachowane_piksele = 0
        piksele_1509_razem = 0

        with rasterio.open(sciezka_wyj, "w", **prof) as dst:
            for r in range(0, ref.height, PASEK):
                h = min(PASEK, ref.height - r)
                okno_ref = Window(0, r, ref.width, h)
                dst_transform = ref.window_transform(okno_ref)

                # 1. Wczytanie paska z 01W1509.tif i sprawdzenie, gdzie dokładnie ma piksele
                dane_ref = ref.read(window=okno_ref)
                maska_gdal_ref = ref.dataset_mask(window=okno_ref)
                istnieje_w_1509 = czy_piksel_istnieje(
                    dane_ref, maska_gdal_ref, ref.nodata, piksel_tla_ref, PROG_TLA
                )
                piksele_1509_razem += int(np.count_nonzero(istnieje_w_1509))

                # Jeśli w tym pasku 01W1509.tif jest samo tło, od razu zapisujemy puste tło
                if not np.any(istnieje_w_1509):
                    pusty_pasek = np.broadcast_to(
                        wypelniacz_tla[:, None, None], (src.count, h, ref.width)
                    )
                    dst.write(pusty_pasek, window=okno_ref)
                    continue

                # 2. Wyznaczenie odpowiadającego fragmentu w przycinanym pliku
                left, bottom, right, top = rasterio.windows.bounds(okno_ref, ref.transform)
                if ref.crs != src.crs:
                    left, bottom, right, top = transform_bounds(
                        ref.crs, src.crs, left, bottom, right, top, densify_pts=21
                    )

                win_src = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
                c0 = max(0, int(np.floor(win_src.col_off)) - 2)
                r0 = max(0, int(np.floor(win_src.row_off)) - 2)
                c1 = min(src.width, int(np.ceil(win_src.col_off + win_src.width)) + 2)
                r1 = min(src.height, int(np.ceil(win_src.row_off + win_src.height)) + 2)

                dane_dopasowane = np.broadcast_to(
                    wypelniacz_tla[:, None, None], (src.count, h, ref.width)
                ).copy()

                if c1 > c0 and r1 > r0:
                    okno_src = Window(c0, r0, c1 - c0, r1 - r0)
                    dane_src_okno = src.read(window=okno_src)
                    maska_gdal_src_okno = src.dataset_mask(window=okno_src)
                    src_okno_transform = src.window_transform(okno_src)

                    # Sprawdzenie ważnych pikseli w pliku źródłowym (usunięcie czarnego tła przed przycięciem)
                    istnieje_w_src_okno = czy_piksel_istnieje(
                        dane_src_okno, maska_gdal_src_okno, src.nodata, piksel_tla_src, PROG_TLA
                    ).astype(np.uint8)

                    # Dopasowanie danych i maski ważności pikseli do siatki 01W1509.tif
                    reproject(
                        source=dane_src_okno,
                        destination=dane_dopasowane,
                        src_transform=src_okno_transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=ref.crs,
                        resampling=Resampling.nearest,
                    )

                    istnieje_w_src = np.zeros((h, ref.width), dtype=np.uint8)
                    reproject(
                        source=istnieje_w_src_okno,
                        destination=istnieje_w_src,
                        src_transform=src_okno_transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=ref.crs,
                        resampling=Resampling.nearest,
                    )

                    # 3. Piksel zostaje TYLKO tam, gdzie istnieje w 01W1509.tif ORAZ istniał w pliku źródłowym
                    zostaw = istnieje_w_1509 & (istnieje_w_src == 1)
                    zachowane_piksele += int(np.count_nonzero(zostaw))

                    # Skasowanie wszystkich pozostałych pikseli:
                    # tam gdzie 01W1509 ma tło -> kopiujemy dokładnie piksele tła z 01W1509.tif
                    if src.count == ref.count:
                        dane_dopasowane[:, ~zostaw] = dane_ref[:, ~zostaw]
                        # dla przerw wewnątrz pasów pliku źródłowego wstawiamy wzorzec tła z 01W1509
                        przerwy_src = istnieje_w_1509 & (istnieje_w_src == 0)
                        dane_dopasowane[:, przerwy_src] = wypelniacz_tla[:, None]
                    else:
                        dane_dopasowane[:, ~zostaw] = wypelniacz_tla[:, None]

                dst.write(dane_dopasowane, window=okno_ref)

            dst.descriptions = src.descriptions
            dst.update_tags(**src.tags())

        procent = 100.0 * zachowane_piksele / max(1, piksele_1509_razem)
        print(
            f"  Zapisano {sciezka_wyj.name}: zachowano {zachowane_piksele:,} pikseli "
            f"({procent:.1f}% powierzchni pola z 15.09)"
        )


def main():
    wyniki = Path(FOLDER_WYNIKOW)
    wyniki.mkdir(parents=True, exist_ok=True)

    with rasterio.open(REFERENCJA) as ref:
        piksel_tla_ref = pobierz_wzorzec_tla(ref)
        print(f"Plik wzorcowy: {Path(REFERENCJA).name} ({ref.width}x{ref.height} px, pasm: {ref.count})")
        print(f"Wartość tła w rogu (0,0) pliku 15.09 (pierwsze 3 pasma): {piksel_tla_ref[:3]}")

        for sciezka in map(Path, DO_PRZYCIECIA):
            wyjscie = wyniki / f"{sciezka.stem}_przyciete.tif"
            print(f"Przetwarzanie: {sciezka.name} -> {wyjscie.name} ...")
            dopasuj_i_przytnij(sciezka, ref, piksel_tla_ref, wyjscie)

    print(f"\nGotowe! Pliki wynikowe są w: {wyniki.resolve()}")


if __name__ == "__main__":
    main()