[GlitchLab] Quick Hotfix — registry not loading filters (Unknown filter 'default_identity')
-----------------------------------------------------------------------------
1) Ensure the filters package is imported once at startup, before the UI:
   In 'glitchlab/gui/main.py' (or the launcher), add near the top:

       import glitchlab.filters  # forces registration of all filters

2) Also make sure 'glitchlab/filters/__init__.py' explicitly imports your modules.
   Example MODULES tuple should include:
       ('default_identity', 'anisotropic_contour_warp', 'block_mosh_grid',
        'phase_glitch', 'spectral_shaper', 'depth_displace')

3) Restart the app. In the 'Filter' tab you should now see the list populated.
