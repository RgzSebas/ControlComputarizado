import time

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, ClientsideFunction

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
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


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Control Computarizado"),
            html.H3("Simulador Computacional de Procesos"),
            html.Div(
                id="intro",
                children="",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
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
            ]),

            html.Label('Selección de Entrada'),
                    dcc.Dropdown(
                    id='entrada-dropdown',
                    options=[
                        {'label': 'Escalon', 'value': 'escalon'},
                        {'label': 'Sierra', 'value': 'sierra'}
                    ],
                    value='escalon'  # Default value
                ),
            dcc.Input(id='amp', type='number', placeholder='Amplitud'),

            html.Label('Intervalo de Muestreo'),
            dcc.Slider(
                id='intervalo-slider',
                min=0,
                max=4,
                step=1,
                marks=["0.01s", "0.05s", "0.1s", "0.5s", "1s"],
                value=5  # Default value
            ),

            html.Label('Perturbación'),
            dcc.Input(id='amp_pert', type='number', placeholder='Amplitud'),


            html.Button('EMPEZAR', id='start-button', n_clicks=0),
            html.Button('STOP', id='stop-button', n_clicks=0),

            html.Label('Sebastian Rodriguez Castro - A01700378'),
            ],
    )

# Create a function to generate the graphs
def generate_graphs():
    return html.Div(
        id="graph-card",
        children=[
            dcc.Graph(id='graph1'),  # First graph
            dcc.Graph(id='graph2')   # Second graph
        ]
    )

app.layout = html.Div([

    html.Div([
        description_card(),
        generate_control_card()
    ], className="four columns"),

    # Add the graphs in the second column
    html.Div([
        generate_graphs()
    ], className="eight columns")
])

# Run the server
if __name__ == "__main__":
    app.run_server(debug=False)
