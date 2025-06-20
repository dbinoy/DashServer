# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd

app = Dash()

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


df1 = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
fig1 = px.bar(df1, x="Fruit", y="Amount", color="City", barmode="group")
fig1.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

df2 = pd.read_csv('usa-agricultural-exports-2011.csv')

df3 = pd.read_csv('gdp-life-exp-2007.csv')
fig3 = px.scatter(df3, x="gdp per capita", y="life expectancy",
                 size="population", color="continent", hover_name="country",
                 log_x=True, size_max=60)


df4 = pd.read_csv('gapminderDataFiveYear.csv')

df5 = pd.read_csv('country_indicators.csv')

app.layout = html.Div(
    # style={'backgroundColor': colors['background']}, 
    children=[
        html.H1(children='Hello Dash'),
        html.Br(),
        html.Div(style={'textAlign': 'center','color': colors['text']},
            children='Dash: A web application framework for your data.'
        ),
        html.Br(),
        dcc.Markdown(children=markdown_text),
        html.Br(),

        html.H2("Change the value in the text box to see callbacks in action!"),
        html.Div([
            "Input: ",
            dcc.Input(id='my-input', value='initial value', type='text')
        ]),
        html.Br(),
        html.Div(id='my-output'),

        dcc.Graph(
            id='example-graph',
            figure=fig1
        ),
        html.Br(),
        html.H1(children='US Agriculture Exports (2011)'),        
        generate_table(df2),
        html.Br(),
        html.H1(children='Life Expentacy vs GDP'), 
        dcc.Graph(
            id='life-exp-vs-gdp',
            figure=fig3
        ),   
        html.Br(),
        html.Div([
            html.Div(children=[
                html.Label('Dropdown'),
                dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),

                html.Br(),
                html.Label('Multi-Select Dropdown'),
                dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],
                            ['Montréal', 'San Francisco'],
                            multi=True),

                html.Br(),
                html.Label('Radio Items'),
                dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Label('Checkboxes'),
                dcc.Checklist(['New York City', 'Montréal', 'San Francisco'],
                            ['Montréal', 'San Francisco']
                ),

                html.Br(),
                html.Label('Text Input'),
                dcc.Input(value='MTL', type='text'),

                html.Br(),
                html.Label('Slider'),
                dcc.Slider(
                    min=0,
                    max=9,
                    marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
                    value=5,
                ),
            ], style={'padding': 10, 'flex': 1})
        ], style={'display': 'flex', 'flexDirection': 'row'}),
        
        html.Br(),
        html.H1(children='Graph with Slider'), 
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            df4['year'].min(),
            df4['year'].max(),
            step=None,
            value=df4['year'].min(),
            marks={str(year): str(year) for year in df4['year'].unique()},
            id='year-slider'
        ),
        
        html.Br(),
        html.H1(children='Country Indicator Graph with multiple Input controls'), 
        html.Div([

            html.Div([
                dcc.Dropdown(
                    df5['Indicator Name'].unique(),
                    'Fertility rate, total (births per woman)',
                    id='xaxis-column'
                ),
                dcc.RadioItems(
                    ['Linear', 'Log'],
                    'Linear',
                    id='xaxis-type',
                    inline=True
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    df5['Indicator Name'].unique(),
                    'Life expectancy at birth, total (years)',
                    id='yaxis-column'
                ),
                dcc.RadioItems(
                    ['Linear', 'Log'],
                    'Linear',
                    id='yaxis-type',
                    inline=True
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ]),

        dcc.Graph(id='indicator-graphic'),

        dcc.Slider(
            df5['Year'].min(),
            df5['Year'].max(),
            step=None,
            id='year--slider',
            value=df5['Year'].max(),
            marks={str(year): str(year) for year in df5['Year'].unique()},

        )                        
])

@callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'

@callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = df4[df4.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig

@callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value'),
    Input('year--slider', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df5[df5['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
                     y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
                     hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name,
                     type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    return fig


if __name__ == '__main__':
    app.run(debug=True)
