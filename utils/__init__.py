"""
Utility functions for EKF demonstrations.

Contains helpers for:
- Coolant temperature profiles
- Chart styling and plotting
- Simulation management
"""

from utils.coolant_profiles import generate_coolant_profile
from utils.plotting import economist_chart_style, get_colors

# Get theme-aware colors
COLORS = get_colors()

__all__ = ['generate_coolant_profile', 'economist_chart_style', 'get_colors', 'COLORS']

