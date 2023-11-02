# Sebastian Rodriguez - A01700378

import time

import dash
from dash import dcc
from dash import html

import pathlib

import webbrowser

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Simulador Computacional de Procesos"

server = app.server
app.config.suppress_callback_exceptions = True

# Open the default web browser to the specified URL
url = "http://127.0.0.1:8050/"
time.sleep(2)
webbrowser.open(url)

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Define the layout of the web application
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H5("Control Computarizado"),
            html.H3("Simulador Computacional de Procesos"),
            html.Div(id="intro", children=""),
        ], id="description-card", className="twelve columns"),
    ], className="row", style={'padding': '10px'}),

    html.Div([
        html.Div([
            # Input fields for ARX model coefficients
            html.Label('Coeficientes del modelo ARX'),
            html.Table([
                html.Tr([
                    html.Td(dcc.Input(id='a1', type='number', placeholder='a1')),
                    html.Td(dcc.Input(id='b1', type='number', placeholder='b1'))
                ]),
                html.Tr([
                    html.Td(dcc.Input(id='a2', type='number', placeholder='a2')),
                    html.Td(dcc.Input(id='b2', type='number', placeholder='b2'))
                ]),
                html.Tr([
                    html.Td(dcc.Input(id='a3', type='number', placeholder='a3')),
                    html.Td(dcc.Input(id='b3', type='number', placeholder='b3'))
                ]),
                html.Tr([
                    html.Td(dcc.Input(id='a4', type='number', placeholder='a4')),
                    html.Td(dcc.Input(id='b4', type='number', placeholder='b4'))
                ])
            ], style={'padding': '10px'}),

            # Dropdown for selecting input signal type
            html.Label('Selección de Entrada'),
            dcc.Dropdown(
                id='entrada-dropdown',
                options=[
                    {'label': 'Escalon', 'value': 'escalon'},
                    {'label': 'Sierra', 'value': 'sierra'}
                ],
                value='escalon'  # Default value
            ),

            # Input field for amplitude of input signal
            dcc.Input(id='amp', type='number', placeholder='Amplitud'),

            # Slider for selecting the sampling interval
            html.Label('Intervalo de Muestreo'),
            dcc.Slider(
                id='intervalo-slider',
                min=0,
                max=4,
                step=1,
                marks={"0": "0.01s", "1": "0.05s", "2": "0.1s", "3": "0.5s", "4": "1s"},
                value=5  # Default value
            ),

            # Input field for amplitude of perturbation signal
            html.Label('Perturbación'),
            dcc.Input(id='amp_pert', type='number', placeholder='Amplitud'),

            # Start and stop buttons
            html.Div([
                html.Button('EMPEZAR', id='start-button', n_clicks=0),
                html.Button('STOP', id='stop-button', n_clicks=0),
            ], style={'padding': '10px'}),

            # Student information
            html.Label('Sebastian Rodriguez Castro - A01700378'),

        ], id="control-card", className="four columns", style={'padding': '10px'}),

        html.Div([
            # Graphs for displaying results
            dcc.Graph(id='graph1'),
            dcc.Graph(id='graph2')
        ], className="eight columns", style={'padding': '10px'}),

    ], id="graph-card"),

])

# Run the server
if __name__ == "__main__":
    app.run_server(debug=False)
