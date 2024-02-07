import plotly.graph_objects as go
import numpy as np
import plotly.tools
import plotly.subplots

from models import SEIR_test, SEIRP_test, SEIQR_test, SEIQDRP_test
from models import time

# Create the figure with subplots
fig = go.Figure()
fig.set_subplots(rows=1, cols=2, shared_yaxes=False)
# Add sin(x) subplot
fig.add_trace(go.Scatter(x=time, y=SEIR_test(), mode='lines', name='sin(x)', line=dict(color='blue')))

# Add cos(x) subplot
fig.add_trace(go.Scatter(x=time, y=SEIRP_test(), mode='lines', name='cos(x)', line=dict(color='red')), row=1, col=2)

# Update layout to display subplots
fig.update_layout(
    title='sin(x) and cos(x) with Vertical Stretch',
    xaxis=dict(title='x'),
    yaxis=dict(title='y'),
    xaxis2=dict(title='x'),
    yaxis2=dict(title='y'),
    height=800,
    width=1600,
)

# Add a slider for vertical stretch
fig.update_layout(
    sliders=[
        dict(
            steps=[dict(method='restyle', args=[{'y': [SEIR_test(Xbeta=s)]}], label=f'Vertical Stretch: {s:.1f}') for s in np.linspace(0, 10, 20)],
            active=5,
            currentvalue={'prefix': 'Stretch Factor: '},
            pad={"t": 50}
        )
    ]
)

# Show the figure
fig.show()
