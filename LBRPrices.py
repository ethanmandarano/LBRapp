import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
import calendar
from scipy import stats
import plotly.express as px
from PIL import ImageColor

# Initialize the app with a Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
# server = app.server  # This is for WSGI servers to use

# GitHub data URL
# GITHUB_RAW_URL = "https://raw.githubusercontent.com/ethanmandarano/LBRapp/refs/heads/main/lumber_data.csv"

lumber_data_full = pd.read_csv('lumber_data.csv')
# Load and process the data 
# lumber_data_full = pd.read_csv(GITHUB_RAW_URL)

# Extract descriptions (assuming the first row contains descriptions)
descriptions = lumber_data_full.iloc[0, 1:].to_dict()

# Copy data excluding the first row (which contains descriptions)
lumber_data = lumber_data_full.iloc[1:].copy()

# Parse dates with specified format
lumber_data['Tag'] = pd.to_datetime(lumber_data['Tag'], format='%m/%d/%Y', errors='coerce')

# Drop rows with unparseable dates
lumber_data = lumber_data.dropna(subset=['Tag'])

# Remove duplicate dates
lumber_data = lumber_data.drop_duplicates(subset=['Tag'])

# Ensure the dates are sorted
lumber_data.sort_values('Tag', inplace=True)

# Reset the index if necessary
lumber_data.reset_index(drop=True, inplace=True)


# Get the latest prices and second-to-latest prices
latest_prices = lumber_data.iloc[-1, 1:].to_dict()
previous_prices = lumber_data.iloc[-2, 1:].to_dict()

# Calculate percentage changes
pct_changes = {}
for col in latest_prices:
    if col in previous_prices and previous_prices[col] != 0:
        pct_change = ((float(latest_prices[col]) - float(previous_prices[col])) / float(previous_prices[col])) * 100
        pct_changes[col] = pct_change
    else:
        pct_changes[col] = 0

# Identify the unique types based on last three words
types = set(' '.join(desc.split()[-3:]) for desc in descriptions.values())

products_by_type = {t: [] for t in types}
for col, desc in descriptions.items():
    type_name = ' '.join(desc.split()[-3:])  # Get last three words
    if type_name in types:
        # Format the percentage change with color and arrow
        pct_change = pct_changes.get(col, 0)
        arrow = "↑" if pct_change > 0 else "↓" if pct_change < 0 else "→"
        
        # Define color based on percentage change
        if pct_change > 0:
            color = "green"
        elif pct_change < 0:
            color = "red"
        else:
            color = "grey"
        
        products_by_type[type_name].append({
            'label': html.Div([
                # Description column
                html.Div(
                    desc,
                    style={
                        'display': 'inline-block',
                        'width': '60%',  # Adjust width as needed
                        'padding-right': '10px',
                        'font-size': '13px',
                        'white-space': 'nowrap',
                        'overflow': 'hidden',
                        'text-overflow': 'ellipsis'
                    }
                ),
                # Price column
                html.Div(
                    f"${latest_prices.get(col, 'N/A')}",
                    style={
                        'display': 'inline-block',
                        'width': '20%',  # Adjust width as needed
                        'text-align': 'right',
                        'padding-right': '10px',
                        'font-size': '13px',
                        'white-space': 'nowrap'
                    }
                ),
                # Percentage change column
                html.Div(
                    [
                        html.Span(
                            f"{arrow}{abs(pct_change):.1f}%",
                            style={
                                'color': color,
                                'font-size': '13px',
                                'white-space': 'nowrap'
                            }
                        )
                    ],
                    style={
                        'display': 'inline-block',
                        'width': '20%',  # Adjust width as needed
                        'text-align': 'right'
                    }
                )
            ], style={
                'display': 'flex',
                'align-items': 'center',
                'width': '100%',
                'padding': '4px 8px',
                'border-radius': '4px',
                'background-color': 'rgba(0,0,0,0.02)',
                'margin': '2px 0'
            }),
            'value': col
        })

# Generate the layout components
checklist_components = []
for idx, (t, products) in enumerate(products_by_type.items()):
    checklist_components.extend([
        html.H5(t, style={'font-size': '16px', 'margin-bottom': '4px'}),
        dbc.Checklist(
            id=f'lumber-checklist-{idx}',
            options=products,
            inline=False,
            switch=True,
            style={
                'margin-bottom': '8px',
                'display': 'flex',
                'flex-direction': 'column',
                'gap': '2px',
                'width': '100%'
            }
        ),
        html.Hr(style={'margin': '4px 0'}),
    ])

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Lumber Prices Dashboard"),
            html.Hr(),
            html.Div(checklist_components),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label('Select Date Range:'),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=lumber_data['Tag'].min(),
                        max_date_allowed=lumber_data['Tag'].max(),
                        start_date=lumber_data['Tag'].min(),
                        end_date=lumber_data['Tag'].max(),
                        display_format='Y-MM-DD',
                        style={'width': '100%'}
                    )
                ], width=12),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label('Aggregation Method:'),
                    dcc.Dropdown(
                        id='aggregation-method',
                        options=[
                            {'label': 'Median', 'value': 'median'},
                            {'label': 'Average', 'value': 'mean'}
                        ],
                        value='median',
                        clearable=False
                    )
                ], width=12),
            ]),
            html.Br(),
            dbc.Button("Download Data", id="download-button", color="primary"),
            dcc.Download(id="download-data"),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='lumber-graph'),
        ], width=9),
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Correlation Analysis"),
            dcc.Graph(id='correlation-heatmap')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Volatility Analysis"),
            dcc.Graph(id='volatility-graph')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Seasonality Chart"),
            dcc.Graph(id='seasonality-chart')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Seasonality Table"),
            dash_table.DataTable(
                id='seasonality-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Spread Analysis"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label('Base Product:'),
                        dcc.Dropdown(
                            id='base-product-dropdown',
                            options=[],
                            placeholder='Select base product'
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label('Spread Type:'),
                        dcc.Dropdown(
                            id='spread-type-dropdown',
                            options=[
                                {'label': 'Absolute Spread ($)', 'value': 'absolute'},
                                {'label': 'Percentage Spread (%)', 'value': 'percentage'}
                            ],
                            value='absolute',
                            clearable=False
                        )
                    ], width=6),
                ]),
                dcc.Graph(id='basis-graph'),
                html.Br(),
                html.H4("Spread Statistics"),
                dash_table.DataTable(
                    id='basis-stats-table',
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                )
            ])
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Historical Basis Ranges"),
            dcc.Graph(id='basis-range-box'),
            html.Div([
                html.Label('Time Period:'),
                dcc.Dropdown(
                    id='basis-period-dropdown',
                    options=[
                        {'label': 'Last 30 Days', 'value': '30D'},
                        {'label': 'Last 90 Days', 'value': '90D'},
                        {'label': 'Year to Date', 'value': 'YTD'},
                        {'label': 'Last 12 Months', 'value': '12M'},
                        {'label': 'All Time', 'value': 'ALL'}
                    ],
                    value='12M'
                )
            ])
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("3D Volatility Surface"),
            dcc.Graph(id='3d-price-graph'),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Statistical Summary"),
            dash_table.DataTable(
                id='stats-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ])
    ]),
], fluid=True)

# Callback for downloading data
@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    [State(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def download_data(n_clicks, *args):
    selected_products_groups = args[:-2]
    start_date = args[-2]
    end_date = args[-1]
    if not n_clicks:
        raise PreventUpdate

    # Gather all selected products from all checklists
    selected_products = [product for group in selected_products_groups if group for product in group]

    if not selected_products:
        raise PreventUpdate

    df_to_download = lumber_data[(lumber_data['Tag'] >= start_date) & (lumber_data['Tag'] <= end_date)]
    columns_to_include = ['Tag'] + selected_products
    df_to_download = df_to_download[columns_to_include]
    csv_string = df_to_download.to_csv(index=False, encoding='utf-8')

    return dict(content=csv_string, filename="selected_data.csv")

# Callback for updating the main graph
@app.callback(
    Output('lumber-graph', 'figure'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_graph(*args):
    selected_products_groups = args[:-2]
    start_date = args[-2]
    end_date = args[-1]
    selected_products = [product for group in selected_products_groups if group for product in group]
    if not selected_products:
        return {
            'data': [],
            'layout': go.Layout(
                title='Lumber Prices Over Time',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                showlegend=True,
                annotations=[dict(
                    text='Please select at least one product to display the graph.',
                    x=0.5, y=0.5, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14)
                )]
            )
        }

    df = lumber_data[(lumber_data['Tag'] >= start_date) & (lumber_data['Tag'] <= end_date)].copy()
    df.sort_values('Tag', inplace=True)

    traces = []
    for product in selected_products:
        # Original price trace
        traces.append(go.Scatter(
            x=df['Tag'],
            y=df[product],
            mode='lines',
            name=descriptions[product]
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title='Lumber Prices Over Time',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            showlegend=True
        )
    }

# Callback for updating the correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_heatmap(*args):
    selected_products_groups = args[:-2]
    start_date = args[-2]
    end_date = args[-1]
    selected_products = [p for group in selected_products_groups if group for p in group]

    if not selected_products or len(selected_products) < 2:
        return {
            'data': [],
            'layout': go.Layout(
                title='Correlation Heatmap',
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[dict(
                    text='Select at least two products to see the correlation heatmap.',
                    x=0.5, y=0.5, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14)
                )]
            )
        }

    df = lumber_data[(lumber_data['Tag'] >= start_date) & (lumber_data['Tag'] <= end_date)]
    df = df[selected_products].apply(pd.to_numeric, errors='coerce').dropna()
    corr_df = df.corr()

    # Create the heatmap
    heatmap = go.Heatmap(
        z=corr_df.values,
        x=[descriptions[p] for p in corr_df.columns],
        y=[descriptions[p] for p in corr_df.columns],
        colorscale='RdYlGn',
        colorbar=dict(title='Correlation Coefficient'),
        zmin=-1, zmax=1
    )

    return {
        'data': [heatmap],
        'layout': go.Layout(
            title='Correlation Heatmap',
            xaxis={'title': 'Products'},
            yaxis={'title': 'Products'},
            width=1200,  # Adjust the width of the heatmap
            height=600,  # Adjust the height of the heatmap
            margin={'l': 200, 'b': 100, 't': 50, 'r': 50}  # Tweak the margins to fit the heatmap
        )
    }

# Callback for updating the volatility graph
@app.callback(
    Output('volatility-graph', 'figure'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_volatility_graph(*args):
    selected_products_groups = args[:-2]
    start_date = args[-2]
    end_date = args[-1]
    selected_products = [product for group in selected_products_groups if group for product in group]
    if not selected_products:
        return {
            'data': [],
            'layout': go.Layout(
                title='Historical Volatility',
                xaxis={'title': 'Date', 'type': 'date'},
                yaxis={'title': 'Volatility (%)'},
                showlegend=True,
                annotations=[dict(
                    text='Please select at least one product to display the volatility graph.',
                    x=0.5, y=0.5, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14)
                )]
            )
        }

    # Filter data based on date range
    df = lumber_data[(lumber_data['Tag'] >= start_date) & (lumber_data['Tag'] <= end_date)]
    df = df[['Tag'] + selected_products].copy()
    df['Tag'] = pd.to_datetime(df['Tag'])
    df.sort_values('Tag', inplace=True)

    # Convert selected product columns to numeric
    for col in selected_products:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in any of the selected columns
    df.dropna(subset=selected_products, inplace=True)

    traces = []
    for product in selected_products:
        df[f'{product}_returns'] = df[product].pct_change()
        df[f'{product}_volatility'] = df[f'{product}_returns'].rolling(window=5).std() * np.sqrt(252) * 100  # Convert to percentage
        traces.append(go.Scatter(
            x=df['Tag'],
            y=df[f'{product}_volatility'],
            mode='lines',
            name=f"{descriptions[product]} Volatility"
        ))

    figure = {
        'data': traces,
        'layout': go.Layout(
            title='Historical Volatility',
            xaxis={'title': 'Date', 'type': 'date'},
            yaxis={'title': 'Volatility (%)'},
            showlegend=True
        )
    }
    return figure

# Callback for updating the seasonality chart
@app.callback(
    Output('seasonality-chart', 'figure'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('aggregation-method', 'value')  # Include aggregation method
)
def update_seasonality_chart(*args):
    selected_products_groups = args[:-3]
    start_date = args[-3]
    end_date = args[-2]
    agg_method = args[-1]

    # Gather all selected products
    selected_products = [product for group in selected_products_groups if group for product in group]

    if not selected_products:
        # Return an empty figure with a message
        return {
            'data': [],
            'layout': go.Layout(
                title='Seasonality Chart',
                xaxis={'title': 'Month'},
                yaxis={'title': 'Average Price'},
                annotations=[dict(
                    text='Please select at least one product to display the seasonality chart.',
                    x=0.5, y=0.5, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14)
                )]
            )
        }

    # Filter data based on date range
    df = lumber_data.copy()
    df = df[(df['Tag'] >= start_date) & (df['Tag'] <= end_date)]
    df['Month'] = df['Tag'].dt.month

    traces = []
    for product in selected_products:
        # Convert product data to numeric
        df[product] = pd.to_numeric(df[product], errors='coerce')

        product_df = df[['Month', product]].dropna()
        if product_df.empty:
            continue

        # Use the selected aggregation method
        monthly_avg = product_df.groupby('Month')[product].agg(agg_method)

        # Ensure months are in correct order
        monthly_avg = monthly_avg.reindex(range(1, 13))

        traces.append(go.Scatter(
            x=[calendar.month_name[m] for m in monthly_avg.index],
            y=monthly_avg.values,
            mode='lines+markers',
            name=descriptions[product]
        ))

    figure = {
        'data': traces,
        'layout': go.Layout(
            title=f'Seasonality Chart - Monthly {agg_method.capitalize()} Prices',
            xaxis={'title': 'Month'},
            yaxis={'title': 'Price'},
            showlegend=True
        )
    }
    return figure


# Callback for updating the seasonality table
@app.callback(
    Output('seasonality-table', 'data'),
    Output('seasonality-table', 'columns'),
    Output('seasonality-table', 'style_data_conditional'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('aggregation-method', 'value')
)
def update_seasonality_table(*args):
    try:
        selected_products_groups = args[:-3]
        start_date = args[-3]
        end_date = args[-2]
        agg_method = args[-1]

        selected_products = [product for group in selected_products_groups if group for product in group]
        if not selected_products:
            return [], [], []

        # Filter data
        df = lumber_data.copy()
        df = df[(df['Tag'] >= start_date) & (df['Tag'] <= end_date)]
        df['Year'] = df['Tag'].dt.year
        df['Month'] = df['Tag'].dt.month

        data_list = []
        style_data_conditional = []

        for product in selected_products:
            df[product] = pd.to_numeric(df[product], errors='coerce')
            product_df = df[['Year', 'Month', product]].dropna()

            product_df['prev_value'] = product_df[product].shift(1)
            product_df['pct_change'] = (
                (product_df[product] - product_df['prev_value']) 
                / product_df['prev_value']
            ) * 100

            grouped = product_df.groupby(['Year', 'Month'])['pct_change'].agg(agg_method).reset_index()
            pivot_df = grouped.pivot(index='Year', columns='Month', values='pct_change')

            for year in pivot_df.index:
                row = {
                    'Product': descriptions[product],
                    'Year': int(year)
                }
                for month in range(1, 13):
                    value = pivot_df.get(month, {}).get(year)
                    if pd.notnull(value):
                        row[calendar.month_name[month]] = f"{value:.2f}%"
                        
                        # Add conditional styling for this cell
                        style_data_conditional.append({
                            'if': {
                                'filter_query': f'{{Product}} = "{descriptions[product]}" && {{Year}} = {int(year)} && {{{calendar.month_name[month]}}} = "{value:.2f}%"',
                                'column_id': calendar.month_name[month]
                            },
                            'color': 'green' if value > 0 else 'red' if value < 0 else 'black'
                        })
                    else:
                        row[calendar.month_name[month]] = "N/A"
                data_list.append(row)

        # Build the columns
        columns = [
            {'name': 'Product', 'id': 'Product'},
            {'name': 'Year', 'id': 'Year'},
        ]
        for month in range(1, 13):
            columns.append({
                'name': calendar.month_name[month],
                'id': calendar.month_name[month]
            })

        return data_list, columns, style_data_conditional

    except Exception as e:
        print(f"Error in update_seasonality_table: {str(e)}")
        return [], [], []


# Callback for updating the statistical summary table
# (Use the updated function from the previous fix)
@app.callback(
    Output('stats-table', 'data'),
    Output('stats-table', 'columns'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_stats_table(*args):
    # (Code as updated previously to fix the error)
    selected_products_groups = args[:-2]
    start_date = args[-2]
    end_date = args[-1]
    selected_products = [p for group in selected_products_groups if group for p in group]

    if not selected_products:
        return [], []

    # Ensure 'LCBM' is included for beta calculation, and remove duplicates
    columns_to_include = ['Tag'] + list(set(selected_products + ['LCBM']))
    df = lumber_data[(lumber_data['Tag'] >= start_date) & (lumber_data['Tag'] <= end_date)]
    df = df[columns_to_include].apply(pd.to_numeric, errors='coerce').dropna()

    stats_list = []
    for product in selected_products:
        product_series = df[product]
        lcbm_series = df['LCBM']

        # Ensure product_series is a Series
        if isinstance(product_series, pd.DataFrame):
            product_series = product_series.iloc[:, 0]

        # Align the series
        common_index = product_series.index.intersection(lcbm_series.index)
        product_series = product_series.loc[common_index]
        lcbm_series = lcbm_series.loc[common_index]

        # Compute statistics
        mean = product_series.mean()
        median = product_series.median()
        std_dev = product_series.std()
        min_price = product_series.min()
        max_price = product_series.max()
        price_range = max_price - min_price
        skewness = product_series.skew()
        kurtosis = product_series.kurtosis()

        if pd.notnull(mean) and mean != 0:
            coef_variation = (std_dev / mean) * 100
        else:
            coef_variation = np.nan

        # Calculate beta vs LCBM and correlation
        if product != 'LCBM':
            covariance = np.cov(product_series, lcbm_series)[0][1]
            variance_lcbm = lcbm_series.var()
            if variance_lcbm != 0:
                beta = covariance / variance_lcbm
            else:
                beta = np.nan
            correlation = product_series.corr(lcbm_series)
        else:
            beta = 1.0
            correlation = 1.0

        stats_list.append({
            'Product': descriptions[product],
            'Mean Price': round(mean, 2),
            'Median Price': round(median, 2),
            'Min Price': round(min_price, 2),
            'Max Price': round(max_price, 2),
            'Price Range': round(price_range, 2),
            'Standard Deviation': round(std_dev, 2) if pd.notnull(std_dev) else np.nan,
            'Coefficient of Variation (%)': round(coef_variation, 2) if pd.notnull(coef_variation) else np.nan,
            'Skewness': round(skewness, 2) if pd.notnull(skewness) else np.nan,
            'Kurtosis': round(kurtosis, 2) if pd.notnull(kurtosis) else np.nan,
            'Beta vs LCBM': round(beta, 2) if pd.notnull(beta) else np.nan,
            'Correlation with LCBM': round(correlation, 2) if pd.notnull(correlation) else np.nan
        })

    # Create DataFrame
    stats_df = pd.DataFrame(stats_list)
    columns = [{'name': col, 'id': col} for col in stats_df.columns]
    data = stats_df.to_dict('records')

    return data, columns

# Callback for updating the 3D price visualization
@app.callback(
    Output('3d-price-graph', 'figure'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_3d_graph(*args):
    selected_products_groups = args[:-2]
    start_date = args[-2]
    end_date = args[-1]
    selected_products = [p for group in selected_products_groups if group for p in group]

    if not selected_products or len(selected_products) < 1:
        return {
            'data': [],
            'layout': go.Layout(
                title='3D Volatility Surface',
                scene=dict(
                    xaxis_title='Date',
                    yaxis_title='Products',
                    zaxis_title='Volatility (%)'
                ),
                annotations=[dict(
                    text='Select at least one product to see the 3D volatility surface.',
                    x=0.5, y=0.5, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14)
                )]
            )
        }

    # Prepare data
    df = lumber_data[(lumber_data['Tag'] >= start_date) & (lumber_data['Tag'] <= end_date)]
    df = df[['Tag'] + selected_products].copy()
    df.sort_values('Tag', inplace=True)
    df['Tag'] = pd.to_datetime(df['Tag'])
    df.set_index('Tag', inplace=True)
    df.dropna(inplace=True)

    # Calculate rolling volatility for each product
    volatility_df = pd.DataFrame()
    for product in selected_products:
        df[product] = pd.to_numeric(df[product], errors='coerce')
        returns = df[product].pct_change()
        volatility = returns.rolling(window=5).std() * np.sqrt(252) * 100  # Annualized volatility
        volatility_df[product] = volatility

    volatility_df.dropna(inplace=True)

    # Create meshgrid for dates and products
    dates = volatility_df.index
    products = selected_products

    date_indices = {date: idx for idx, date in enumerate(dates)}
    product_indices = {product: idx for idx, product in enumerate(products)}

    Z = np.zeros((len(products), len(dates)))

    for p_idx, product in enumerate(products):
        Z[p_idx] = volatility_df[product].values

    data = [
        go.Surface(
            z=Z,
            x=dates,
            y=[descriptions[p] for p in products],
            colorscale='Viridis'
        )
    ]

    figure = {
        'data': data,
        'layout': go.Layout(
            title='3D Volatility Surface Over Time and Products',
            scene=dict(
                xaxis=dict(title='Date', type='date'),
                yaxis=dict(title='Products'),
                zaxis=dict(title='Volatility (%)')
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
    }
    return figure

# Callback to update base product dropdown options
@app.callback(
    Output('base-product-dropdown', 'options'),
    Output('base-product-dropdown', 'value'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))]
)
def update_base_product_options(*selected_products_groups):
    selected_products = [p for group in selected_products_groups if group for p in group]
    
    if not selected_products:
        return [], None
        
    options = [{'label': descriptions[p], 'value': p} for p in selected_products]
    return options, selected_products[0]

# Callback for basis graph and statistics
@app.callback(
    Output('basis-graph', 'figure'),
    Output('basis-stats-table', 'data'),
    Output('basis-stats-table', 'columns'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('base-product-dropdown', 'value'),
    Input('spread-type-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_basis_analysis(*args):
    selected_products_groups = args[:-4]
    base_product = args[-4]
    spread_type = args[-3]
    start_date = args[-2]
    end_date = args[-1]
    
    selected_products = [p for group in selected_products_groups if group for p in group]
    
    if not selected_products or not base_product or len(selected_products) < 2:
        empty_fig = {
            'data': [],
            'layout': go.Layout(
                title='Spread Analysis',
                annotations=[dict(
                    text='Please select at least two products and a base product.',
                    x=0.5, y=0.5,
                    xref='paper', yref='paper',
                    showarrow=False
                )]
            )
        }
        return empty_fig, [], []

    # Filter data based on date range
    df = lumber_data[(lumber_data['Tag'] >= start_date) & 
                     (lumber_data['Tag'] <= end_date)].copy()
    
    # Calculate spreads
    traces = []
    stats_data = []
    
    base_price = pd.to_numeric(df[base_product], errors='coerce')
    
    for product in selected_products:
        if product != base_product:
            product_price = pd.to_numeric(df[product], errors='coerce')
            
            if spread_type == 'absolute':
                spread = product_price - base_price
                spread_label = 'Spread ($)'
            else:  # percentage
                spread = ((product_price - base_price) / base_price) * 100
                spread_label = 'Spread (%)'
            
            # Create trace for this spread
            traces.append(go.Scatter(
                x=df['Tag'],
                y=spread,
                mode='lines',
                name=f'{descriptions[product]} vs {descriptions[base_product]}'
            ))
            
            # Calculate statistics
            stats_data.append({
                'Product Pair': f'{descriptions[product]} vs {descriptions[base_product]}',
                'Average Spread': f"{spread.mean():.2f}{'%' if spread_type == 'percentage' else '$'}",
                'Min Spread': f"{spread.min():.2f}{'%' if spread_type == 'percentage' else '$'}",
                'Max Spread': f"{spread.max():.2f}{'%' if spread_type == 'percentage' else '$'}",
                'Std Dev': f"{spread.std():.2f}{'%' if spread_type == 'percentage' else '$'}",
                'Current Spread': f"{spread.iloc[-1]:.2f}{'%' if spread_type == 'percentage' else '$'}"
            })
    
    # Create the figure
    figure = {
        'data': traces,
        'layout': go.Layout(
            title=f'Spread Analysis ({spread_type.capitalize()} Spread)',
            xaxis={'title': 'Date'},
            yaxis={'title': spread_label},
            showlegend=True,
            hovermode='x unified'
        )
    }
    
    # Create columns for the stats table
    columns = [
        {'name': col, 'id': col} for col in [
            'Product Pair', 'Average Spread', 'Min Spread', 
            'Max Spread', 'Std Dev', 'Current Spread'
        ]
    ]
    
    return figure, stats_data, columns

# Add this callback for the basis range analysis
@app.callback(
    Output('basis-range-box', 'figure'),
    [Input(f'lumber-checklist-{idx}', 'value') for idx in range(len(types))],
    Input('base-product-dropdown', 'value'),
    Input('spread-type-dropdown', 'value'),
    Input('basis-period-dropdown', 'value')
)
def update_basis_range_analysis(*args):
    selected_products_groups = args[:-3]
    base_product = args[-3]
    spread_type = args[-2]
    period = args[-1]
    
    selected_products = [p for group in selected_products_groups if group for p in group]
    
    if not selected_products or not base_product or len(selected_products) < 2:
        return {'data': [], 'layout': {'title': 'Select products to view basis ranges'}}

    # Calculate date range based on period selection
    end_date = pd.Timestamp.now()
    if period == '30D':
        start_date = end_date - pd.Timedelta(days=30)
    elif period == '90D':
        start_date = end_date - pd.Timedelta(days=90)
    elif period == 'YTD':
        start_date = pd.Timestamp(end_date.year, 1, 1)
    elif period == '12M':
        start_date = end_date - pd.Timedelta(days=365)
    else:  # ALL
        start_date = lumber_data['Tag'].min()

    df = lumber_data[(lumber_data['Tag'] >= start_date)].copy()
    base_price = pd.to_numeric(df[base_product], errors='coerce')
    
    box_data = []
    z_scores = []
    
    for product in selected_products:
        if product != base_product:
            product_price = pd.to_numeric(df[product], errors='coerce')
            
            if spread_type == 'absolute':
                spread = product_price - base_price
                suffix = '$'
            else:
                spread = ((product_price - base_price) / base_price) * 100
                suffix = '%'
            
            # Calculate z-score of current spread
            current_spread = spread.iloc[-1]
            mean_spread = spread.mean()
            std_spread = spread.std()
            z_score = (current_spread - mean_spread) / std_spread if std_spread != 0 else 0
            
            box_data.append(go.Box(
                y=spread,
                name=f"{descriptions[product]}",
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            z_scores.append({
                'Product': descriptions[product],
                'Current': f"{current_spread:.2f}{suffix}",
                'Mean': f"{mean_spread:.2f}{suffix}",
                'Z-Score': f"{z_score:.2f}",
                'Percentile': f"{stats.percentileofscore(spread.dropna(), current_spread):.1f}%"
            })

    fig = {
        'data': box_data,
        'layout': go.Layout(
            title=f'Basis Distribution ({period})',
            yaxis={'title': f'Spread ({suffix})'},
            showlegend=True,
            annotations=[
                go.layout.Annotation(
                    x=i,
                    y=z['Current'].rstrip(suffix),
                    text=f"Current: {z['Current']}<br>Z: {z['Z-Score']}<br>%ile: {z['Percentile']}",
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                ) for i, z in enumerate(z_scores)
            ]
        )
    }
    
    return fig

if __name__ == '__main__':
     app.run_server(debug=True, host='0.0.0.0', port=8050)