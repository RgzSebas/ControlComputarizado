# Sebastian Rodriguez - A01700378

import time
import numpy as np

import dash
from dash import dcc, html
import dash_daq as daq
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import webbrowser

# Define global variables
current_mode = 'Manual'  # 'Manual' or 'Automatic'
# Add other global variables for model parameters and state here

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

input_fig = go.Figure(
    data=[go.Scatter(x=[0], y=[0], mode='lines+markers')],
    layout=go.Layout(title='Señal de Entrada', xaxis=dict(title='Tiempo (s)'), yaxis=dict(title='Magnitud'),
                     margin=dict(l=30, r=20, t=30, b=20))
)

output_fig = go.Figure(
    data=[go.Scatter(x=[0], y=[0], mode='lines+markers')],
    layout=go.Layout(title='Salida del Sistema', xaxis=dict(title='Tiempo (s)'), yaxis=dict(title='Magnitud'),
                     margin=dict(l=30, r=20, t=30, b=20))
)

# Define the layout of the web application
app.layout = html.Div([


    html.Div([
        html.Div([
            html.Div([
                    html.Div([
                        html.H5("Control Computarizado"),
                        html.H3(children="Simulador Computacional de Procesos", style={"margin": "0px"}),
                        html.Div(id="intro", children=""),
                    ], id="description-card", className="twelve columns"),
                ], className="row", style={'padding': '0px'}),
            html.Div(id='feedback', children='', style={'color': 'red'}),
            # Input fields for ARX model coefficients
            html.Hr(),
            daq.ToggleSwitch(
                id='mode-switch',
                label='Manual - Automatico',
                labelPosition='bottom'),
            html.Hr(),
            #html.Div(id='mode-switch-output'),
            html.Label('Coeficientes del modelo ARX (Requiere minimo a1)'),
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

            html.Div(id='auto-mode-inputs', children=[
                html.Label('Setpoint & PID'),
                # First row with Setpoint and Kp
                html.Div([
                    html.Div(dcc.Input(id='setpoint', type='number', placeholder='Setpoint'),
                             style={'flex': '50%', 'padding': '5px'}),
                    html.Div(dcc.Input(id='Kp', type='number', placeholder='Kp'),
                             style={'flex': '50%', 'padding': '5px'})
                ], style={'display': 'flex', 'width': '100%'}),

                # Second row with Ki and Kd
                html.Div([
                    html.Div(dcc.Input(id='Ki', type='number', placeholder='Ki'),
                             style={'flex': '50%', 'padding': '5px'}),
                    html.Div(dcc.Input(id='Kd', type='number', placeholder='Kd'),
                             style={'flex': '50%', 'padding': '5px'})
                ], style={'display': 'flex', 'width': '100%'})
            ], style={'padding': '10px'}),

            # Dropdown for selecting input signal type
            html.Div(id='manual-mode-inputs', children=[
                html.Label('Entrada'),
                html.Div([
                    # Dropdown for selecting input signal type
                    dcc.Dropdown(
                        id='entrada-dropdown',
                        options=[
                            {'label': 'Escalon', 'value': 'escalon'},
                            {'label': 'Sierra', 'value': 'sierra'}
                        ],
                        value='escalon',  # Default value
                        style={'flex': '1'}  # Flex property for equal spacing
                    ),

                    # Input field for amplitude of input signal
                    dcc.Input(
                        id='amp',
                        type='number',
                        placeholder='Amplitud',
                        style={'flex': '1', 'marginLeft': '10px'}  # marginLeft for spacing
                    )
                ], style={'display': 'flex', 'padding': '10px'}),  # Flex display for horizontal row
            ], style={'padding': '10px'}),

            html.Hr(),
            # Slider for selecting the sampling interval
            html.Label('Intervalo de Muestreo'),
            dcc.Slider(
                id='intervalo-slider',
                min=0,
                max=4,
                step=1,
                marks={"0": "0.1s", "1": "0.25s", "2": "0.5s", "3": "1s", "4": "10s"},
                value=1  # Default value
            ),

            html.Hr(),
            # Input field for amplitude of perturbation signal
            html.Label('Perturbación'),
            dcc.Input(id='amp_pert', type='number', placeholder='Amplitud', value=0),

            html.Hr(),
            # Start and stop buttons
            html.Div([
                # START button with green color
                html.Button('START', id='start-button', n_clicks=0,
                            style={'marginRight': '5px', 'backgroundColor': 'green', 'color': 'white'}),

                # STOP button with red color
                html.Button('STOP', id='stop-button', n_clicks=0,
                            style={'marginRight': '5px', 'backgroundColor': 'red', 'color': 'white'}),

                # RESET button with blue color
                html.Button('RESET', id='reset-button', n_clicks=0,
                            style={'backgroundColor': 'blue', 'color': 'white'}),

                # Invisible div for triggering updates
                html.Div(id='start-stop-trigger', style={'display': 'none'}),
            ], style={'display': 'flex', 'padding': '10px'}),

            # Student information
            html.H5('Sebastian Rodriguez Castro - A01700378'),

        ], id="control-card", className="four columns", style={'padding': '10px'}),

        html.Div([
            # Graphs for displaying results
            dcc.Graph(id='graph_input', figure=input_fig),
            dcc.Graph(id='graph_output', figure=output_fig),
            dcc.Interval(
                    id='interval-component',
                    interval=1*1000,  # in milliseconds (update every 1 second)
                    n_intervals=0
                ),
        ], className="eight columns", style={'padding': '10px'}),

    ], id="graph-card"),

])

# Callback to toggle the visibility of input fields based on the mode
@app.callback(
    [Output('auto-mode-inputs', 'style'), Output('manual-mode-inputs', 'style')],
    [Input('mode-switch', 'value')]
)
def toggle_input_fields(mode_switch_value):
    if mode_switch_value:  # Automatic mode
        return {'display': 'block'}, {'display': 'none'}
    else:  # Manual mode
        return {'display': 'none'}, {'display': 'block'}

# Callback for the Toggle Switch
@app.callback(
    Output('mode-switch-output', 'children'),  # Update this with actual output component
    Input('mode-switch', 'value')
)
def update_mode(switch_value):
    global current_mode
    current_mode = 'Automatico' if switch_value else 'Manual'
    return current_mode

# Global variable to track if the simulation should run
is_simulation_running = False

# Callback to start or stop the simulation and provide feedback
@app.callback(
    [Output('start-stop-trigger', 'children'), Output('feedback', 'children')],
    [Input('start-button', 'n_clicks'), Input('stop-button', 'n_clicks')],
    [State('mode-switch', 'value'), State('entrada-dropdown', 'value'), State('amp', 'value'),
     State('amp_pert', 'value'), State('a1', 'value'), State('setpoint', 'value'),
     State('Kp', 'value'), State('Ki', 'value'), State('Kd', 'value')],  # Add states for Kp, Ki, Kd
    prevent_initial_call=True
)
def control_simulation(start_clicks, stop_clicks, mode_switch, entrada_type, amp, amp_pert, a1, setpoint, Kp, Ki, Kd):
    global is_simulation_running
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    feedback_message = ""

    if 'start-button' in changed_id:
        missing_inputs = []
        # Check required inputs based on the mode
        if mode_switch:  # Automatic mode
            if a1 is None: missing_inputs.append("Minimo a1 (coeficiente ARX)")
            if setpoint is None: missing_inputs.append("Setpoint")
            if None in [Kp, Ki, Kd]: missing_inputs.append("Coeficientes PID")
        else:  # Manual mode
            if entrada_type is None: missing_inputs.append("Tipo de entrada")
            if amp is None: missing_inputs.append("Amplitud de entrada")
            if a1 is None: missing_inputs.append("Minimo a1 (coeficiente ARX)")

        if missing_inputs:
            feedback_message = "Campos requeridos: " + ", ".join(missing_inputs)
            return "", feedback_message

        is_simulation_running = True
    elif 'stop-button' in changed_id:
        is_simulation_running = False

    return "", feedback_message


# Callback to adjust the update interval based on the slider value
@app.callback(
    Output('interval-component', 'interval'),
    [Input('intervalo-slider', 'value')],
)
def update_interval(value):
    # Map the slider value to an actual time interval in milliseconds
    intervals = {0: 100, 1: 250, 2: 500, 3: 1000, 4: 10000}  # These values are in milliseconds
    return intervals.get(value, 1000)  # Default to 1 second (1000 ms) if value is not in the dictionary


@app.callback(
    [Output('graph_input', 'figure'), Output('graph_output', 'figure')],
    [Input('interval-component', 'n_intervals'), Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks'), Input('reset-button', 'n_clicks')],
    [State('a1', 'value'), State('a2', 'value'), State('a3', 'value'), State('a4', 'value'),
     State('b1', 'value'), State('b2', 'value'), State('b3', 'value'), State('b4', 'value'),
     State('entrada-dropdown', 'value'), State('amp', 'value'), State('amp_pert', 'value'),
     State('mode-switch', 'value'), State('setpoint', 'value'), State('intervalo-slider', 'value'),
     State('Kp', 'value'), State('Ki', 'value'), State('Kd', 'value')],  # Add PID coefficient states
    prevent_initial_call=True
)
def update_and_reset_graphs(n_intervals, start_n_clicks, stop_n_clicks, reset_n_clicks,
                            a1, a2, a3, a4, b1, b2, b3, b4, entrada_type, amp, amp_pert, mode_switch, setpoint, intervalo_slider_value,
                            Kp, Ki, Kd):  # Add PID coefficients as parameters
    global y, u, step, is_simulation_running

    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-button':
        # Reset logic
        y = [0]
        u = [0]
        step = 0
        is_simulation_running = False
        # Create empty figures for input and output
        input_fig = go.Figure(
            layout=go.Layout(title='Input Signal', xaxis=dict(title='Time'), yaxis=dict(title='Input Value')))
        output_fig = go.Figure(
            layout=go.Layout(title='System Output', xaxis=dict(title='Time'), yaxis=dict(title='Output Value')))
        return input_fig, output_fig

    elif trigger_id == 'interval-component' and is_simulation_running:
        # Check if all required inputs are provided
        if None in [a1]:
            raise dash.exceptions.PreventUpdate

        # Map the slider value to an actual time interval
        interval_map = {0: 0.1, 1: 0.25, 2: 0.5, 3: 1, 4: 10}
        dt = interval_map.get(intervalo_slider_value, 1)  # Default to 1 second

        # Update the ARX coefficients and current mode based on user inputs
        a = [a1, a2, a3, a4]
        b = [b1, b2, b3, b4]
        current_mode = 'Automatico' if mode_switch else 'Manual'

        if current_mode == 'Automatico':
            # Use user-provided PID parameters
            # Check if PID parameters are provided
            # if None in [Kp, Ki, Kd]:
            #    raise dash.exceptions.PreventUpdate

            # PID controller for automatic mode
            u_t = pid_controller(setpoint, y[-1], Kp, Ki, Kd, dt)
            y_t = arx_step(y, u, a, b, d)
            y.append(y_t)
            u.append(u_t)
            step += 1
        else:
            # Generate input signal based on dropdown selection
            if entrada_type == 'escalon':
                u_t = amp
            elif entrada_type == 'sierra':
                u_t = (step % 1) * amp  # Example of sawtooth input
            else:
                u_t = 0  # Default value

            # Advance the simulation by one step
            y_t = arx_step(y, u, a, b, d)
            y.append(y_t)
            u.append(u_t)
            step += 1

        # Generate time array based on the number of data points and the sampling interval
        time_array = np.arange(0, len(u) * dt, dt)[:len(u)]

        # Update the graph data to use time_array for the x-axis
        input_fig = go.Figure(
            data=[go.Scatter(x=time_array, y=u, mode='lines+markers')],
            layout=go.Layout(title='Señal de Entrada', xaxis=dict(title='Tiempo (s)'), yaxis=dict(title='Magnitud'),
                             margin=dict(l=30, r=20, t=30, b=20))
        )

        output_fig = go.Figure(
            data=[go.Scatter(x=time_array, y=y, mode='lines+markers')],
            layout=go.Layout(title='Salida del Sistema', xaxis=dict(title='Tiempo (s)'), yaxis=dict(title='Magnitud'),
                             margin=dict(l=30, r=20, t=30, b=20))
        )

        return input_fig, output_fig

    # Return existing figures if no update or reset is required
    return dash.no_update

# Initialize global variables for the simulation
y = [0]  # Output of the system
u = [0]  # Input to the system
a = [0.5, 0.2]  # ARX model output coefficients
b = [1, 0.5]  # ARX model input coefficients
d = 1  # Discrete dead time
step = 0  # Current simulation step


def arx_step(y, u, a_coeffs, b_coeffs, d):
    """
    Calculates the next step in the ARX model.

    :param y: List of previous output values of the system.
    :param u: List of input values to the system.
    :param a_coeffs: List of coefficients for output values (a1, a2, ...). None if not provided.
    :param b_coeffs: List of coefficients for input values (b1, b2, ...). None if not provided.
    :param d: Discrete dead time.
    :return: Next output value of the system.
    """
    y_t = 0

    # Handle a_coeffs
    for i, a in enumerate(a_coeffs):
        if a is not None and len(y) > i:
            y_t -= a * y[-i - 1]

    # Handle b_coeffs
    for j, b in enumerate(b_coeffs):
        if b is not None and len(u) > d + j:
            y_t += b * u[-d - j - 1]

    return y_t


def pid_controller(set_point, current_value, Kp, Ki, Kd, dt):
    error = set_point - current_value
    integral = error * dt
    derivative = (error - (set_point - current_value)) / dt
    return Kp * error + Ki * integral + Kd * derivative

def auto_tune_pid(set_point, output):
    # Basic heuristic for PID tuning
    # This is a simple example and should be replaced with a more sophisticated method
    Kp = 0.2 * set_point
    Ki = 0.05 * set_point
    Kd = 0.1 * set_point
    return Kp, Ki, Kd




# Run the server
if __name__ == "__main__":
    app.run_server(debug=False)
