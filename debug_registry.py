from glitchlab.core.registry import available, describe
import glitchlab.filters  # ważne: powoduje import __init__ i rejestrację filtrów

print("Zarejestrowane filtry:", ", ".join(available()))
for name in available():
    print("\n---", name, "---")
    print(describe(name))
