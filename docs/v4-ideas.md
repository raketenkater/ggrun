# v4 Ideas

Ideas parked for later design work. These are not committed product behavior.

## Recommender quant switching

- Explore left/right arrow controls on recommended model rows to switch the
  selected model between available quantizations.
- Use this only if it stays clear that the default recommendation is still the
  safest hardware-fit choice.
- Open questions:
  - Should left/right switch within the selected model only, or also expose a
    compact quant picker?
  - Should categories keep their policy caps, e.g. Fastest stays Q4-capped,
    while Smartest can expose Q5/Q8/BF16?
  - How do we avoid users picking a quant that fits on paper but makes the PC
    unusable without enough RAM/VRAM reserve?
