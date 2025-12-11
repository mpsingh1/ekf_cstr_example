"""
Plotting utilities with Economist magazine styling.

Provides consistent, professional chart styling across the application.
Supports both light and dark themes.
"""

import streamlit as st


def get_colors():
    """
    Get color palette based on current Streamlit theme.
    Uses Streamlit's actual theme colors for backgrounds.
    
    Returns:
    --------
    dict : Color dictionary with theme-appropriate colors
    """
    try:
        theme_base = st.get_option("theme.base")
        bg_color = st.get_option("theme.backgroundColor")
        secondary_bg = st.get_option("theme.secondaryBackgroundColor")
        text_color = st.get_option("theme.textColor")
        is_dark = theme_base == "dark"
    except:
        is_dark = False
        bg_color = "#ffffff"
        secondary_bg = "#f0f2f6"
        text_color = "#1a1a1a"
    
    if is_dark:
        # Dark mode: use lighter colors for visibility
        return {
            'true_state': '#b0b0b0',  # Light gray for dark mode
            'measurements': '#ff8c00',  # Orange (same)
            'estimate': '#ff4444',  # Brighter red for dark mode
            'model': '#4caf50',  # Brighter green
            'coolant': '#26a69a',  # Brighter teal
            'ci_fill': 'rgba(255, 68, 68, 0.15)',  # Brighter red fill
            'text': text_color,
            'background': bg_color,  # Use Streamlit's actual background
            'plot_bg': bg_color,  # Pure black for charts in dark mode
            'grid': text_color,
            'legend_bg': bg_color,
            'legend_border': '#555',
            'sidebar_bg': secondary_bg
        }
    else:
        return {
            'true_state': '#4a4a4a',
            'measurements': '#ff8c00',  # Orange
            'estimate': '#E3120B',  # Red
            'model': '#2e7d32',
            'coolant': '#00897b',
            'ci_fill': 'rgba(227, 18, 11, 0.08)',  # Light red
            'text': text_color,
            'background': bg_color,  # Use Streamlit's actual background
            'plot_bg': bg_color,
            'grid': text_color,
            'legend_bg': bg_color,
            'legend_border': '#d0d0d0',
            'sidebar_bg': secondary_bg
        }


# Note: Don't set COLORS at module level - it needs to refresh on each Streamlit rerun
# Always call get_colors() dynamically


def economist_chart_style(fig, title="", height=500):
    """
    Apply Economist magazine styling to Plotly charts.
    Automatically adapts to light/dark theme.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to style
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Styled figure
    """
    colors = get_colors()
    
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color=colors['text']),
        title=dict(text=title, font=dict(size=16, color=colors['text']), x=0, xanchor='left'),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['plot_bg'],
        height=height,
        margin=dict(l=50, r=10, t=30, b=120),  # Optimized: left margin for y-axis label, reduced top
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,  # Further below chart to avoid x-axis label
            xanchor="center",
            x=0.5,
            bgcolor=colors['legend_bg'],
            bordercolor=colors['legend_border'],
            borderwidth=1,
            font=dict(size=10, color=colors['text'])
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showline=True,
        linewidth=2,
        linecolor=colors['text'],
        mirror=False,
        title_font=dict(size=11, color=colors['text']),
        title_standoff=15  # Extra space between axis and title
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showline=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor=colors['text'],
        title_font=dict(size=11, color=colors['text'])
    )
    
    return fig

