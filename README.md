# glitchlab — *glitch as controlled ontology*  

**glitchlab** to narzędzie glitch-art, które pozwala kontrolować **lokalizację**, **intensywność** i **metodę** rozprzestrzeniania błędów w obrazach `.png` / `.jpg`.  
Domyślnie stara się **chronić tekst i krawędzie**, a także umożliwia celowanie w symboliczne regiony za pomocą masek.  


## ⚡ Instalacja

```code
git clone https://github.com/you/glitchlab
cd glitchlab
pip install -r requirements.txt
```

---

## 🚀 Użycie

Minimalny przykład:

```bash
python -m glitchlab.gui.main -i input.png -o output.png --preset default
```

Z wybranym presetem:

```bash
python -m glitchlab.gui.main -i input.png -o glitch.png --preset focus_text
```

Z własnym configiem:

```bash
python -m glitchlab.gui.main -i input.png -o glitch.png --config my_config.yaml
```

---

## 🎨 Presety

* `default.yaml` — bazowy glitch, zachowuje czytelność tekstu.
* `focus_text.yaml` — mocniejsze glitchowanie tła, tekst wyraźny.
* `hard_glitch.yaml` — wysokie przesunięcia RGB, mocny datamosh.
* `ultra_fantasy.yaml` — połączenie efektów **wave distortion**, **pixel sorting**, **mask inverts** i **channel shuffle** → surrealistyczny efekt.

---

## 🔧 Własne filtry

Nowe filtry dodajesz w katalogu `glitchlab/filters/`, np.:

```python
from ..core.registry import register

@register("my_filter")
def my_filter(arr, ctx, strength=1.0):
    # arr = numpy array obrazu
    # ctx = kontekst (maski, amplitude, meta)
    return arr
```

Po czym rejestrujesz w `filters/__init__.py`.

---
