# Register filters on import

from .noise import *
from .posterize import *
from .rgb_offset import *
from .scanlines import *
from .stripes import *
from .block_mosh import *
from .protect_edges import *

# filtry importowane jako moduły (bo same rejestrują się przez @register)
from . import amp_mask_mul
from . import pixel_sort
from . import wave_distort
from . import channel_shuffle
from . import color_invert_masked
from . import depth_displace
from . import depth_parallax
from . import nazca_lines
