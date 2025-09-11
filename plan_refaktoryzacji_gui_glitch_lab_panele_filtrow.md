# Plan refaktoryzacji GUI GlitchLab – panele filtrów

Poniżej zapisuję krok po kroku:

1. Inwentaryzacja aktualnego GUI (pliki, klasy, gdzie podpinamy filtry).
2. Warstwa rejestru paneli (registry) – jeden punkt rejestracji panelu do filtra.
3. Wspólny kontrakt panelu: `PanelBase` → `build(form)`, `to_params()`, `apply(img, ctx)`.
4. Panele dedykowane: `DepthDisplacePanel`, `DepthParallaxPanel`, `AnisotropicContourWarpPanel`, `RgbOffsetPanel`.
5. Integracja w `app.py`/`main.py`: dropdown z listą filtrów, dynamiczne osadzanie panelu, przycisk „Dodaj do pipeline” i „Zastosuj do podglądu”.
6. Rejestracja paneli i mapowanie nazw filtra → klasa panelu.
7. (Opcja) Diagnostyka/metyki: sidecar z histogramem, gęstością krawędzi i podglądami masek/amp.

Po potwierdzeniu będę generować gotowe pliki: `gui/panels/base.py`, `gui/panels/registry.py`, `gui/panels/depth_displace.py`, `gui/panels/depth_parallax.py`, `gui/panels/anisotropic_contour_warp.py`, `gui/panels/rgb_offset.py`, oraz modyfikację `gui/app.py`/`gui/main.py` do podpięcia rejestru.

