"""Tokyo station-level ridership prediction and transit-equity analysis.

Cross-sectional planning model for the Tokyo rail network (474 stations across
the 23 special wards). The package exposes one feature-computation path shared
by training and serving (the anti-skew boundary), a spatially-blocked
evaluation harness, and a thin FastAPI inference service.

See ``IMPLEMENTATION_PLAN.md`` for the phase ordering and ``CLAUDE.md`` for the
architecture rules that must not be violated.
"""

from __future__ import annotations

__version__ = "0.0.0"
