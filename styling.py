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