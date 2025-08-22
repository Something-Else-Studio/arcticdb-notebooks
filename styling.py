# styling.py

def style_table(df, hide_index=True, max_rows=20):
    """
    Applies a custom theme to a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to style.
        hide_index (bool): Whether to hide the DataFrame index.
        max_rows (int): Maximum number of rows to display.

    Returns:
        Styler object: A styled DataFrame object ready for display.
    """
    # Limit rows first if specified
    if max_rows is not None:
        df = df.head(max_rows)
    
    # Define your theme colors here
    theme = {
        'header_bg': '#141C52',
        'header_font': '#F9F9F9',
        'cell_bg': '#75D0E8',
        'cell_font': '#141C52',
        'hover_bg': '#783ABB',
        'hover_font': '#F9F9F9'
    }

    # Define the styles
    styles = [
        {'selector': 'th', # Style the table headers
         'props': [('background-color', theme['header_bg']),
                   ('color', theme['header_font']),
                   ('font-weight', 'bold'),
                   ('text-align', 'left'),
                   ('padding', '10px')]},
        {'selector': 'td', # Style the data cells
         'props': [('background-color', theme['cell_bg']),
                   ('color', theme['cell_font']),
                   ('padding', '8px')]},
        {'selector': 'tr:hover td', # Style rows on hover
         'props': [('background-color', theme['hover_bg']),
                   ('color', theme['hover_font'])]}
    ]

    # Apply the styles
    styled_df = df.style.set_table_styles(styles)

    if hide_index:
        styled_df = styled_df.hide(axis="index")
        
    # You can add more general formatting here, e.g., for numbers
    # styled_df = styled_df.format('{:,.2f}', subset=df.select_dtypes(include=['number']).columns)

    return styled_df


def export_table_html(styled_df, filename, title="Sports Data Table"):
    """Export a styled DataFrame to a standalone HTML file"""
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .table-container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
            }}
            h1 {{
                color: #141C52;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="table-container">
            <h1>{title}</h1>
            {styled_df.to_html()}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_string)
    print(f"Exported: {filename}")


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")


def setup_chart_styling():
    """
    Set up seaborn and matplotlib with custom styling based on the table theme.
    
    Returns:
        dict: Color palette and styling options
    """
    # Define the theme colors (same as in style_table)
    theme = {
        'header_bg': '#141C52',
        'header_font': '#F9F9F9',
        'cell_bg': '#75D0E8',
        'cell_font': '#141C52',
        'hover_bg': '#783ABB',
        'hover_font': '#F9F9F9'
    }
    
    # Create a color palette using Figma theme colors
    palette_colors = [
        '#75D0E8',            # SECONDARY/SKY BLUE - primary
        '#141C52',            # PRIMARY/DARK BLUE - secondary  
        '#783ABB',            # PRIMARY/NEON PURPLE - accent
        '#F9F9F9',            # PRIMARY/OFF WHITE - neutral
        '#DDE5ED',            # SECONDARY/COOL GREY - light gray
        '#EEF2F6',            # SECONDARY/COOL GREY 50 - lighter gray
        '#43D6AC',            # PRIMARY/NEON GREEN - complementary
        '#A1EBD6',            # PRIMARY/NEON GREEN 50 - lighter green
        '#103392'             # SECONDARY/DEEP BLUE - deeper blue variant
    ]
    
    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'axes.edgecolor': theme['header_bg'],
        'axes.linewidth': 1.2,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.8
    })
    
    # Set the color palette
    sns.set_palette(palette_colors)
    
    # Configure matplotlib defaults
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': theme['header_bg'],
        'axes.linewidth': 1.2,
        'axes.labelcolor': theme['header_bg'],
        'axes.titlecolor': theme['header_bg'],
        'text.color': theme['header_bg'],
        'xtick.color': theme['header_bg'],
        'ytick.color': theme['header_bg'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    return {
        'theme': theme,
        'palette': palette_colors,
        'primary_color': theme['cell_bg'],
        'secondary_color': theme['header_bg'],
        'accent_color': theme['hover_bg'],
        'text_color': theme['header_bg'],
        'light_text': theme['header_font']
    }


def style_catplot(grid, colors=None):
    """
    Apply custom styling to a seaborn catplot (FacetGrid).
    
    Args:
        grid: seaborn FacetGrid object from catplot
        colors: optional color scheme dict from setup_chart_styling()
    """
    if colors is None:
        colors = setup_chart_styling()
    
    # Style each subplot in the grid
    for ax in grid.axes.flat:
        # Set background and spine colors
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color(colors['secondary_color'])
            spine.set_linewidth(1.2)
        
        # Style tick labels
        ax.tick_params(colors=colors['text_color'], labelsize=9)
        
        # Style axis labels
        ax.set_xlabel(ax.get_xlabel(), color=colors['text_color'], fontsize=10)
        ax.set_ylabel(ax.get_ylabel(), color=colors['text_color'], fontsize=10)
        
        # Style grid
        ax.grid(True, color='#E0E0E0', linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
    
    # Style the overall figure
    grid.fig.patch.set_facecolor('white')
    
    return grid


def style_lmplot(grid, colors=None):
    """
    Apply custom styling to a seaborn lmplot (FacetGrid).
    
    Args:
        grid: seaborn FacetGrid object from lmplot
        colors: optional color scheme dict from setup_chart_styling()
    """
    if colors is None:
        colors = setup_chart_styling()
    
    # Style each subplot in the grid
    for ax in grid.axes.flat:
        # Set background and spine colors
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color(colors['secondary_color'])
            spine.set_linewidth(1.2)
        
        # Style tick labels and axes
        ax.tick_params(colors=colors['text_color'], labelsize=9)
        ax.set_xlabel(ax.get_xlabel(), color=colors['text_color'], fontsize=10)
        ax.set_ylabel(ax.get_ylabel(), color=colors['text_color'], fontsize=10)
        
        # Style grid
        ax.grid(True, color='#E0E0E0', linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
    
    # Style the overall figure
    grid.fig.patch.set_facecolor('white')
    
    return grid


def style_matplotlib_chart(ax=None, colors=None, title=None):
    """
    Apply custom styling to a matplotlib axes object.
    
    Args:
        ax: matplotlib axes object (if None, uses current axes)
        colors: optional color scheme dict from setup_chart_styling()
        title: optional title for the chart
    """
    if colors is None:
        colors = setup_chart_styling()
    
    if ax is None:
        ax = plt.gca()
    
    # Set background and spine colors
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color(colors['secondary_color'])
        spine.set_linewidth(1.2)
    
    # Style tick labels and axes
    ax.tick_params(colors=colors['text_color'], labelsize=9)
    ax.set_xlabel(ax.get_xlabel(), color=colors['text_color'], fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), color=colors['text_color'], fontsize=10)
    
    # Add title if provided
    if title:
        ax.set_title(title, color=colors['text_color'], fontsize=12, fontweight='bold')
    
    # Style grid
    ax.grid(True, color='#E0E0E0', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Ensure figure background is white
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    
    return ax


def setup_plotly_theme():
    """
    Set up Plotly with custom styling based on the table theme.
    
    Returns:
        dict: Plotly template and color configuration
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Define the theme colors (same as in setup_chart_styling)
    theme = {
        'header_bg': '#141C52',
        'header_font': '#F9F9F9',
        'cell_bg': '#75D0E8',
        'cell_font': '#141C52',
        'hover_bg': '#783ABB',
        'hover_font': '#F9F9F9'
    }
    
    # Figma color palette
    palette_colors = [
        '#75D0E8',            # SECONDARY/SKY BLUE - primary
        '#141C52',            # PRIMARY/DARK BLUE - secondary  
        '#783ABB',            # PRIMARY/NEON PURPLE - accent
        '#F9F9F9',            # PRIMARY/OFF WHITE - neutral
        '#DDE5ED',            # SECONDARY/COOL GREY - light gray
        '#EEF2F6',            # SECONDARY/COOL GREY 50 - lighter gray
        '#43D6AC',            # PRIMARY/NEON GREEN - complementary
        '#A1EBD6',            # PRIMARY/NEON GREEN 50 - lighter green
        '#103392'             # SECONDARY/DEEP BLUE - deeper blue variant
    ]
    
    # Create custom Plotly template
    custom_template = go.layout.Template(
        layout=go.Layout(
            colorway=palette_colors,
            font=dict(color=theme['header_bg'], size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='#E0E0E0',
                linecolor=theme['header_bg'],
                tickcolor=theme['header_bg'],
                title_font=dict(color=theme['header_bg'], size=14),
                tickfont=dict(color=theme['header_bg'], size=11)
            ),
            yaxis=dict(
                gridcolor='#E0E0E0',
                linecolor=theme['header_bg'],
                tickcolor=theme['header_bg'],
                title_font=dict(color=theme['header_bg'], size=14),
                tickfont=dict(color=theme['header_bg'], size=11)
            ),
            title=dict(
                font=dict(color=theme['header_bg'], size=16),
                x=0.05,
                xanchor='left'
            ),
            legend=dict(
                font=dict(color=theme['header_bg'], size=11),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=theme['header_bg'],
                borderwidth=1
            ),
            hoverlabel=dict(
                bgcolor='white',
                bordercolor=theme['header_bg'],
                font=dict(color=theme['header_bg'])
            )
        )
    )
    
    # Register the template
    pio.templates["custom_theme"] = custom_template
    pio.templates.default = "custom_theme"
    
    return {
        'theme': theme,
        'palette': palette_colors,
        'template': custom_template,
        'primary_color': theme['cell_bg'],
        'secondary_color': theme['header_bg'],
        'accent_color': theme['hover_bg'],
        'text_color': theme['header_bg'],
        'light_text': theme['header_font']
    }


def create_plotly_catplot(data, x_col, y_col, hue_col=None, col_col=None, title=None, 
                         col_order=None, col_wrap=None, sharey=True, kind="bar", 
                         error_col=None, add_value_labels=False):
    """
    Create an interactive Plotly version of seaborn catplot.
    
    Args:
        data: DataFrame with the data
        x_col: Column for x-axis
        y_col: Column for y-axis values
        hue_col: Column for color grouping
        col_col: Column for faceting (subplots)
        title: Main title for the plot
        col_order: Order of facet columns
        col_wrap: Number of columns before wrapping to next row
        sharey: Whether subplots share y-axis
        kind: Type of plot ("bar", "point", "box", etc.)
        error_col: Column name for error bars
        add_value_labels: Whether to add value labels on bars
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Setup theme
    colors = setup_plotly_theme()
    
    if col_col:
        # Create faceted plot
        unique_cols = data[col_col].unique() if col_order is None else col_order
        n_cols = len(unique_cols)
        
        # Calculate grid dimensions
        if col_wrap is not None:
            n_plot_cols = min(col_wrap, n_cols)
            n_plot_rows = (n_cols + col_wrap - 1) // col_wrap  # Ceiling division
        else:
            n_plot_cols = n_cols
            n_plot_rows = 1
        
        # Create subplots with proper grid layout
        fig = make_subplots(
            rows=n_plot_rows, cols=n_plot_cols,
            subplot_titles=[str(col) for col in unique_cols],
            shared_yaxes=sharey,
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        for i, col_val in enumerate(unique_cols):
            col_data = data[data[col_col] == col_val]
            
            # Calculate row and column for this subplot
            if col_wrap is not None:
                subplot_row = (i // col_wrap) + 1
                subplot_col = (i % col_wrap) + 1
            else:
                subplot_row = 1
                subplot_col = i + 1
            
            if hue_col:
                # Group by hue column
                for j, hue_val in enumerate(col_data[hue_col].unique()):
                    hue_data = col_data[col_data[hue_col] == hue_val]
                    color = colors['palette'][j % len(colors['palette'])]
                    
                    if kind == "bar":
                        bar_kwargs = {
                            'x': hue_data[x_col],
                            'y': hue_data[y_col],
                            'name': str(hue_val),
                            'marker_color': color,
                            'showlegend': (i == 0),  # Only show legend for first subplot
                            'hovertemplate': f"{hue_col}: {hue_val}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                        }
                        
                        # Add error bars if specified
                        if error_col and error_col in hue_data.columns:
                            bar_kwargs['error_y'] = dict(
                                type='data',
                                array=hue_data[error_col],
                                visible=True,
                                color=colors['accent_color'],
                                thickness=2,
                                width=3
                            )
                        
                        # Add value labels if specified
                        if add_value_labels:
                            bar_kwargs['text'] = hue_data[y_col]
                            bar_kwargs['textposition'] = 'outside'
                            bar_kwargs['texttemplate'] = '%{text:.2f}'
                        
                        fig.add_trace(go.Bar(**bar_kwargs), row=subplot_row, col=subplot_col)
            else:
                # Single series per facet
                if kind == "bar":
                    bar_kwargs = {
                        'x': col_data[x_col],
                        'y': col_data[y_col],
                        'name': str(col_val),
                        'marker_color': colors['palette'][i % len(colors['palette'])],
                        'showlegend': False,
                        'hovertemplate': f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                    }
                    
                    # Add error bars if specified
                    if error_col and error_col in col_data.columns:
                        bar_kwargs['error_y'] = dict(
                            type='data',
                            array=col_data[error_col],
                            visible=True,
                            color=colors['accent_color'],
                            thickness=2,
                            width=3
                        )
                    
                    # Add value labels if specified
                    if add_value_labels:
                        bar_kwargs['text'] = col_data[y_col]
                        bar_kwargs['textposition'] = 'outside'
                        bar_kwargs['texttemplate'] = '%{text:.2f}'
                    
                    fig.add_trace(go.Bar(**bar_kwargs), row=subplot_row, col=subplot_col)
    else:
        # Single plot
        fig = go.Figure()
        
        if hue_col:
            # Group by hue column
            for i, hue_val in enumerate(data[hue_col].unique()):
                hue_data = data[data[hue_col] == hue_val]
                color = colors['palette'][i % len(colors['palette'])]
                
                if kind == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=hue_data[x_col],
                            y=hue_data[y_col],
                            name=str(hue_val),
                            marker_color=color,
                            hovertemplate=f"{hue_col}: {hue_val}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                        )
                    )
        else:
            # Single series
            if kind == "bar":
                fig.add_trace(
                    go.Bar(
                        x=data[x_col],
                        y=data[y_col],
                        marker_color=colors['primary_color'],
                        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                    )
                )
    
    # Update layout with dynamic height for wrapped plots
    if col_col and col_wrap is not None:
        plot_height = max(400, n_plot_rows * 300 + 100)  # Dynamic height based on rows
    else:
        plot_height = 500
    
    fig.update_layout(
        title=title,
        height=plot_height,
        margin=dict(t=80, b=80, l=80, r=80),
        template="custom_theme",
        showlegend=False if col_col else True  # No legend for faceted plots
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    
    return fig


def create_plotly_scatterplot(data, x_col, y_col, hue_col=None, title=None, 
                             show_regression=True, error_bars=None, text_col=None, 
                             xlim=None, ylim=None):
    """
    Create an interactive Plotly scatter plot with optional regression line.
    
    Args:
        data: DataFrame with the data
        x_col: Column for x-axis
        y_col: Column for y-axis
        hue_col: Column for color grouping
        title: Title for the plot
        show_regression: Whether to show regression line
        error_bars: Dict with x/y error bar column names
        text_col: Column for text labels on data points
        xlim: Tuple of (min, max) for x-axis limits
        ylim: Tuple of (min, max) for y-axis limits
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Setup theme
    colors = setup_plotly_theme()
    
    fig = go.Figure()
    
    if hue_col:
        # Group by hue column
        for i, hue_val in enumerate(data[hue_col].unique()):
            hue_data = data[data[hue_col] == hue_val]
            color = colors['palette'][i % len(colors['palette'])]
            
            # Add scatter points
            scatter_kwargs = {
                'x': hue_data[x_col],
                'y': hue_data[y_col],
                'mode': 'markers+text' if text_col else 'markers',
                'name': str(hue_val),
                'marker': dict(color=color, size=8),
                'hovertemplate': f"{hue_col}: {hue_val}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
            }
            
            # Add text labels if specified
            if text_col and text_col in hue_data.columns:
                scatter_kwargs['text'] = hue_data[text_col]
                scatter_kwargs['textposition'] = 'top center'
                scatter_kwargs['textfont'] = dict(color=colors['text_color'], size=9)
            
            # Add error bars if specified
            if error_bars:
                if 'x' in error_bars:
                    scatter_kwargs['error_x'] = dict(type='data', array=hue_data[error_bars['x']], visible=True)
                if 'y' in error_bars:
                    scatter_kwargs['error_y'] = dict(type='data', array=hue_data[error_bars['y']], visible=True)
            
            fig.add_trace(go.Scatter(**scatter_kwargs))
            
            # Add regression line if requested
            if show_regression:
                try:
                    import numpy as np
                    from sklearn.linear_model import LinearRegression
                    
                    # Fit regression
                    X = hue_data[x_col].values.reshape(-1, 1)
                    y = hue_data[y_col].values
                    reg = LinearRegression().fit(X, y)
                    
                    # Create prediction line
                    x_range = np.linspace(hue_data[x_col].min(), hue_data[x_col].max(), 100)
                    y_pred = reg.predict(x_range.reshape(-1, 1))
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f"{hue_val} trend",
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate="Regression line<extra></extra>",
                        showlegend=False
                    ))
                except ImportError:
                    print("sklearn not available for regression lines")
    else:
        # Single series
        scatter_kwargs = {
            'x': data[x_col],
            'y': data[y_col],
            'mode': 'markers+text' if text_col else 'markers',
            'marker': dict(color=colors['primary_color'], size=8),
            'hovertemplate': f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
            'showlegend': False
        }
        
        # Add text labels if specified
        if text_col and text_col in data.columns:
            scatter_kwargs['text'] = data[text_col]
            scatter_kwargs['textposition'] = 'top center'
            scatter_kwargs['textfont'] = dict(color=colors['text_color'], size=9)
        
        # Add error bars if specified
        if error_bars:
            if 'x' in error_bars:
                scatter_kwargs['error_x'] = dict(type='data', array=data[error_bars['x']], visible=True)
            if 'y' in error_bars:
                scatter_kwargs['error_y'] = dict(type='data', array=data[error_bars['y']], visible=True)
        
        fig.add_trace(go.Scatter(**scatter_kwargs))
        
        # Add regression line if requested
        if show_regression:
            try:
                import numpy as np
                from sklearn.linear_model import LinearRegression
                
                # Fit regression
                X = data[x_col].values.reshape(-1, 1)
                y = data[y_col].values
                reg = LinearRegression().fit(X, y)
                
                # Create prediction line
                x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                y_pred = reg.predict(x_range.reshape(-1, 1))
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    line=dict(color=colors['secondary_color'], width=2),
                    hovertemplate="Regression line<extra></extra>",
                    showlegend=False
                ))
            except ImportError:
                print("sklearn not available for regression lines")
    
    # Update layout
    layout_kwargs = {
        'title': title,
        'xaxis_title': x_col,
        'yaxis_title': y_col,
        'height': 500,
        'margin': dict(t=80, b=80, l=80, r=80),
        'template': "custom_theme"
    }
    
    # Set axis limits if specified
    if xlim:
        layout_kwargs['xaxis'] = dict(range=xlim)
    if ylim:
        layout_kwargs['yaxis'] = dict(range=ylim)
    
    fig.update_layout(**layout_kwargs)
    
    return fig


def create_plotly_histplot(data, x_col, hue_col=None, col_col=None, title=None,
                          col_order=None, col_wrap=None, bins=30, adaptive_bins=True):
    """
    Create an interactive Plotly histogram plot with optional faceting.
    
    Args:
        data: DataFrame with the data
        x_col: Column for histogram values
        hue_col: Column for color grouping (not typically used for histograms)
        col_col: Column for faceting (subplots)
        title: Main title for the plot
        col_order: Order of facet columns
        col_wrap: Number of columns before wrapping to next row
        bins: Number of histogram bins (or default if adaptive_bins=True)
        adaptive_bins: If True, adjust bin count based on data range for each facet
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Setup theme
    colors = setup_plotly_theme()
    
    if col_col:
        # Create faceted histogram plot
        unique_cols = data[col_col].unique() if col_order is None else col_order
        n_cols = len(unique_cols)
        
        # Calculate grid dimensions
        if col_wrap is not None:
            n_plot_cols = min(col_wrap, n_cols)
            n_plot_rows = (n_cols + col_wrap - 1) // col_wrap  # Ceiling division
        else:
            n_plot_cols = n_cols
            n_plot_rows = 1
        
        # Create subplots with proper grid layout
        fig = make_subplots(
            rows=n_plot_rows, cols=n_plot_cols,
            subplot_titles=[str(col) for col in unique_cols],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        for i, col_val in enumerate(unique_cols):
            col_data = data[data[col_col] == col_val]
            
            # Calculate row and column for this subplot
            if col_wrap is not None:
                subplot_row = (i // col_wrap) + 1
                subplot_col = (i % col_wrap) + 1
            else:
                subplot_row = 1
                subplot_col = i + 1
            
            # Calculate optimal bins for this league if adaptive
            if adaptive_bins:
                data_range = col_data[x_col].max() - col_data[x_col].min()
                if data_range <= 15:  # Low-scoring sports (EPL, NHL)
                    optimal_bins = min(15, int(data_range) + 1)
                elif data_range <= 50:  # Medium-scoring
                    optimal_bins = min(25, int(data_range * 0.5))
                else:  # High-scoring sports
                    optimal_bins = min(30, int(data_range * 0.3))
                league_bins = max(10, optimal_bins)  # Minimum 10 bins
            else:
                league_bins = bins
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=col_data[x_col],
                    nbinsx=league_bins,
                    name=str(col_val),
                    marker_color=colors['primary_color'],
                    showlegend=False,
                    hovertemplate=f"{col_val}<br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ),
                row=subplot_row, col=subplot_col
            )
        
        # Update layout with dynamic height for wrapped plots
        if col_wrap is not None:
            plot_height = max(400, n_plot_rows * 300 + 100)  # Dynamic height based on rows
        else:
            plot_height = 500
    else:
        # Single histogram
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=data[x_col],
                nbinsx=bins,
                marker_color=colors['primary_color'],
                hovertemplate=f"Range: %{{x}}<br>Count: %{{y}}<extra></extra>"
            )
        )
        plot_height = 500
    
    # Update layout
    fig.update_layout(
        title=title,
        height=plot_height,
        margin=dict(t=80, b=80, l=80, r=80),
        template="custom_theme",
        showlegend=False
    )
    
    # Update axes labels (these can be overridden after function call)
    fig.update_xaxes(title_text="Points Scored")
    fig.update_yaxes(title_text="Frequency")
    
    return fig


def create_plotly_lmplot(data, x_col, y_col, hue_col=None, col_col=None, title=None, 
                        show_regression=True, aspect=1.0):
    """
    Create an interactive Plotly version of seaborn lmplot with faceting.
    
    Args:
        data: DataFrame with the data
        x_col: Column for x-axis
        y_col: Column for y-axis
        hue_col: Column for color grouping
        col_col: Column for faceting (subplots)
        title: Main title for the plot
        show_regression: Whether to show regression lines
        aspect: Aspect ratio for subplots
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Setup theme
    colors = setup_plotly_theme()
    
    if col_col:
        # Create faceted plot
        unique_cols = sorted(data[col_col].unique())
        n_cols = len(unique_cols)
        
        # Calculate subplot dimensions
        subplot_height = int(500 * aspect)
        total_width = max(800, n_cols * 300)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=n_cols,
            subplot_titles=[str(col) for col in unique_cols],
            horizontal_spacing=0.1
        )
        
        for i, col_val in enumerate(unique_cols):
            col_data = data[data[col_col] == col_val]
            
            if hue_col:
                # Group by hue column
                for j, hue_val in enumerate(col_data[hue_col].unique()):
                    hue_data = col_data[col_data[hue_col] == hue_val]
                    color = colors['palette'][j % len(colors['palette'])]
                    
                    # Add scatter points
                    fig.add_trace(
                        go.Scatter(
                            x=hue_data[x_col],
                            y=hue_data[y_col],
                            mode='markers',
                            name=str(hue_val),
                            marker=dict(color=color, size=6),
                            showlegend=(i == 0),  # Only show legend for first subplot
                            hovertemplate=f"{hue_col}: {hue_val}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                        ),
                        row=1, col=i+1
                    )
                    
                    # Add regression line if requested
                    if show_regression:
                        try:
                            import numpy as np
                            from sklearn.linear_model import LinearRegression
                            
                            if len(hue_data) > 1:
                                # Fit regression
                                X = hue_data[x_col].values.reshape(-1, 1)
                                y = hue_data[y_col].values
                                reg = LinearRegression().fit(X, y)
                                
                                # Create prediction line
                                x_range = np.linspace(hue_data[x_col].min(), hue_data[x_col].max(), 50)
                                y_pred = reg.predict(x_range.reshape(-1, 1))
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_range,
                                        y=y_pred,
                                        mode='lines',
                                        line=dict(color=color, width=2),
                                        hovertemplate="Regression line<extra></extra>",
                                        showlegend=False
                                    ),
                                    row=1, col=i+1
                                )
                        except ImportError:
                            print("sklearn not available for regression lines")
            else:
                # Single series per facet
                color = colors['palette'][i % len(colors['palette'])]
                
                fig.add_trace(
                    go.Scatter(
                        x=col_data[x_col],
                        y=col_data[y_col],
                        mode='markers',
                        marker=dict(color=color, size=6),
                        name=str(col_val),
                        showlegend=False,
                        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                    ),
                    row=1, col=i+1
                )
                
                # Add regression line if requested
                if show_regression:
                    try:
                        import numpy as np
                        from sklearn.linear_model import LinearRegression
                        
                        if len(col_data) > 1:
                            # Fit regression
                            X = col_data[x_col].values.reshape(-1, 1)
                            y = col_data[y_col].values
                            reg = LinearRegression().fit(X, y)
                            
                            # Create prediction line
                            x_range = np.linspace(col_data[x_col].min(), col_data[x_col].max(), 50)
                            y_pred = reg.predict(x_range.reshape(-1, 1))
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=y_pred,
                                    mode='lines',
                                    line=dict(color='grey', width=2),
                                    hovertemplate="Regression line<extra></extra>",
                                    showlegend=False
                                ),
                                row=1, col=i+1
                            )
                    except ImportError:
                        print("sklearn not available for regression lines")
        
        # Update layout for faceted plot
        fig.update_layout(
            title=title,
            height=subplot_height,
            width=total_width,
            margin=dict(t=100, b=80, l=80, r=80),
            template="custom_theme"
        )
        
    else:
        # Single plot (non-faceted)
        fig = create_plotly_scatterplot(data, x_col, y_col, hue_col, title, show_regression)
    
    # Update axes labels
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col)
    
    return fig


def export_plot_html(fig, filename, title="Interactive Chart"):
    """
    Export a Plotly figure to a standalone HTML file that can be embedded in websites.
    
    Args:
        fig: Plotly figure object
        filename: Output filename (should end with .html)
        title: Title for the HTML page
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Configure Plotly to create a standalone HTML file
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
    
    # Use Plotly's built-in HTML export with custom styling
    html_content = fig.to_html(
        include_plotlyjs='cdn',
        config=config,
        div_id="plotly-div"
    )
    
    # Wrap in our custom styling
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .chart-container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
            }}
            .chart-container h1 {{
                color: #141C52;
                margin-bottom: 20px;
                text-align: center;
            }}
            #plotly-div {{
                width: 100%;
                height: 500px;
            }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <h1>{title}</h1>
            {html_content}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_string)
    print(f"Exported: {filename}")


def create_plotly_bar_chart(data, x_col, y_col, title=None, error_col=None, 
                           add_value_labels=False):
    """
    Create a simple interactive Plotly bar chart.
    
    Args:
        data: DataFrame with the data
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis (values)
        title: Title for the plot
        error_col: Column name for error bars
        add_value_labels: Whether to add value labels on bars
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly not available. Install with: pip install plotly")
    
    # Setup theme
    colors = setup_plotly_theme()
    
    # Create bar chart
    fig = go.Figure()
    
    bar_kwargs = {
        'x': data[x_col],
        'y': data[y_col],
        'marker_color': colors['primary_color'],
        'hovertemplate': f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
        'text': data[y_col] if add_value_labels else None,
        'textposition': 'outside' if add_value_labels else None,
        'texttemplate': '%{text:.2f}' if add_value_labels else None
    }
    
    # Add error bars if specified
    if error_col and error_col in data.columns:
        bar_kwargs['error_y'] = dict(
            type='data',
            array=data[error_col],
            visible=True,
            color=colors['accent_color'],
            thickness=2,
            width=3
        )
    
    fig.add_trace(go.Bar(**bar_kwargs))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
        margin=dict(t=80, b=80, l=80, r=80),
        template="custom_theme",
        showlegend=False
    )
    
    return fig


