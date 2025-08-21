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


def export_all_tables(arctic_lib, output_dir='html_exports', max_rows=50):
    """Export all league tables from ArcticDB to HTML files"""
    import os
    
    # Create exports directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export all league tables
    for league in arctic_lib.list_symbols():
        league_data = arctic_lib.read(league).data
        styled_table = style_table(league_data, max_rows=max_rows)
        
        filename = f"{output_dir}/{league.lower()}_table.html"
        title = f"{league} League Statistics"
        
        export_table_html(styled_table, filename, title)
    
    print(f"\nAll tables exported to '{output_dir}' folder!")


def create_index_page(arctic_lib, output_dir='html_exports'):
    """Create an index HTML page with links to all league tables"""
    import os
    
    leagues = arctic_lib.list_symbols()
    league_links = []
    
    for league in leagues:
        filename = f"{league.lower()}_table.html"
        league_links.append(f'<li><a href="{filename}">{league} League Statistics</a></li>')
    
    links_html = '\n'.join(league_links)
    
    index_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Sports League Statistics</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #141C52;
                text-align: center;
                margin-bottom: 30px;
            }}
            ul {{
                list-style: none;
                padding: 0;
            }}
            li {{
                margin: 15px 0;
                padding: 15px;
                background: #75D0E8;
                border-radius: 4px;
                transition: background-color 0.3s;
            }}
            li:hover {{
                background: #783ABB;
            }}
            a {{
                color: #141C52;
                text-decoration: none;
                font-weight: bold;
                font-size: 18px;
            }}
            li:hover a {{
                color: #F9F9F9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sports League Statistics</h1>
            <ul>
                {links_html}
            </ul>
        </div>
    </body>
    </html>
    """
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print(f"Created index.html in '{output_dir}' folder")
    print("You can now upload the folder to GitHub Pages or any web host!")