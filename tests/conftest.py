"""Shared fixtures. The synthetic portrait is deliberately geometric: every
feature has known coordinates, so a test can assert *where* a color ended up.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

SKIN = (215, 175, 140)
HAIR = (45, 33, 24)
EYE = (40, 30, 25)
SHIRT = (60, 90, 170)
LOGO = (200, 40, 40)
PANTS = (25, 25, 60)


@pytest.fixture
def portrait_bytes() -> bytes:
    """A 384×512 mock portrait matching layout.DEFAULT_BOXES."""
    img = np.full((512, 384, 3), 230, dtype=np.uint8)
    img[220:400, 60:324] = SHIRT
    img[220:400, 150:234] = LOGO
    img[30:220, 120:264] = SKIN
    img[30:80, 120:264] = HAIR
    img[110:126, 148:172] = EYE
    img[110:126, 212:236] = EYE
    img[400:512, 140:244] = PANTS
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()
