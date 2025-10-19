# GlitchLab — Glossary (Spec v1)

**S/H/Z (Δ-składniki):**
- **S** — zmiana strukturalna (AST, topologia modułów).
- **H** — zmiana semantyczna (etykiety, sygnatury, kontrakty).
- **Z** — zmiana parametrów/metryk (np. progi, budżety, wagowanie).

**Δ (delta):** wektor [dS, dH, dZ] ważony według `spec/invariants.yaml` i używany w EGDB/BUS.

**Φ (projekcja):** AST → mozaika; każdy węzeł daje (Sel, Act, Obs) na kafelkach/warstwach.

**Ψ (podnoszenie):** mozaika → AST; reguły modyfikacji planu (parametry, if/for, refaktor).

**I1–I4 (inwarianty):**
- **I1:** spójność typów/nośników danych.
- **I2:** spójność warstw i kontraktów.
- **I3:** lokalność zmian (Δ w deklarowanym scope Φ).
- **I4:** monotoniczność budżetów/metryk celu.
