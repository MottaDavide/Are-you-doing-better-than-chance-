import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from scipy import stats


def f(X,Y) -> float:
    """
    WMAPE function

    Parameters
    ----------
    X : array
        Array of uniformly distributed random variables. Forecast
    Y : array
        Array of uniformly distributed random variables. Sales

    Returns
    -------
    float
        WMAPE(X,Y)
    """
    return np.sum(np.abs(X - Y)) / np.sum(Y)


def evaluation(n_max = 1000, maxiter=1, distribution='uniform' , **parameters):
    """
    Evaluate error values for various distributions.

    Parameters:
    - n_max (int): Maximum value of n for which to compute the error values. Default is 1000.
    - maxiter (int): Number of iterations for each value of n. Default is 1.
    - distribution (str): Type of distribution ('uniform', 'norm', 'exp', 'beta', 'betaprime'). Default is 'uniform'.
    - **parameters: Distribution-specific parameters. 

    Returns:
    - error_values (dict): Computed error values for each n.
    - limit (float): Limit value computed using Monte Carlo simulation.
    """
    error_values = {}
    if distribution == 'uniform':
        try:
            a, b = parameters.values()
        except:
            a, b = 0,100
        for n in tqdm(range(2, n_max + 1)):
            errors_for_n = [
            f(stats.uniform.rvs(a, b, n), stats.uniform.rvs(a, b, n))
            for _ in range(maxiter)
        ]
            error_values[n] = np.mean(errors_for_n)

        # MC simulation
        mean = stats.uniform.mean(a, b)
        x = stats.uniform.rvs(a, b, 100000)
        y = stats.uniform.rvs(a, b, 100000)
        lotus = np.mean(np.abs(x - y))
        limit = lotus/mean

    elif distribution == 'norm':
        try:
            loc, scale = parameters.values()
        except:
            loc, scale = 50,100
        for n in tqdm(range(2, n_max + 1)):
            errors_for_n = [
            f(stats.norm.rvs(loc, scale, n), stats.norm.rvs(loc, scale, n))
            for _ in range(maxiter)
        ]
            error_values[n] = np.mean(errors_for_n)

        # MC simulation
        mean = stats.norm.mean(loc,scale)
        x = stats.norm.rvs(loc,scale, 100000)
        y = stats.norm.rvs(loc,scale, 100000)
        lotus = np.mean(np.abs(x - y))
        limit = lotus/mean

    elif distribution == 'exp':
        try:
            loc, scale = parameters.values()
        except:
            loc, scale = 50,100
        for n in tqdm(range(2, n_max + 1)):
            errors_for_n = [
            f(stats.expon.rvs(loc, scale, n), stats.expon.rvs(loc, scale, n))
            for _ in range(maxiter)
        ]
            error_values[n] = np.mean(errors_for_n)
        # MC simulation
        mean = stats.expon.mean(loc,scale)
        x = stats.expon.rvs(loc,scale, 100000)
        y = stats.expon.rvs(loc,scale, 100000)
        lotus = np.mean(np.abs(x - y))
        limit = lotus/mean
    elif distribution == 'beta':
        try:
            a, b, loc, scale = parameters.values()
        except:
            a, b, loc, scale = 2,3,1,1
        for n in tqdm(range(2, n_max + 1)):
            errors_for_n = [
            f(stats.beta.rvs(a, b, loc, scale, n), stats.beta.rvs(a, b, loc, scale, n))
            for _ in range(maxiter)
        ]
            error_values[n] = np.mean(errors_for_n)   
        # MC simulation
        mean = stats.beta.mean(a, b, loc,scale)
        x = stats.beta.rvs(a, b, loc,scale, 100000)
        y = stats.beta.rvs(a, b, loc,scale, 100000)
        lotus = np.mean(np.abs(x - y))
        limit = lotus/mean

    elif distribution == 'betaprime':
        try:
            a, b, loc, scale = parameters.values()
        except:
            a, b, loc, scale = 2,3,1,1
        for n in tqdm(range(2, n_max + 1)):
            errors_for_n = [
            f(stats.betaprime.rvs(a, b, loc, scale, n), stats.betaprime.rvs(a, b, loc, scale, n))
            for _ in range(maxiter)
        ]
            error_values[n] = np.mean(errors_for_n)

        # MC simulation
        mean = stats.betaprime.mean(a, b, loc,scale)
        x = stats.betaprime.rvs(a, b, loc,scale, 100000)
        y = stats.betaprime.rvs(a, b, loc,scale, 100000)
        lotus = np.mean(np.abs(x - y))
        limit = lotus/mean
    return error_values, limit


def plotting(error_values, limit, distribution='uniform'):
    """
    Plot error values for a given distribution.

    Parameters:
    - error_values (dict): Computed error values for each n.
    - limit (float): Limit value to be displayed on the plot.
    - distribution (str): Type of distribution ('uniform', 'norm', 'exp', 'beta', 'betaprime'). Default is 'uniform'.

    Returns:
    None. Displays the plot.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(error_values.keys()),
            y=list(error_values.values()),
            mode='lines',
            name='f(n)',
            line=dict(color='royalblue', width=2.5),
            hoverinfo='x+y'
        )
    )

    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=min(error_values.keys()),
            x1=max(error_values.keys()),
            y0=limit,
            y1=limit,
            line=dict(color="Red", width=2, dash="dashdot"),
        )
    )
    fig.add_annotation(
        x=max(error_values.keys()),
        y=limit,
        xshift=-50,
        yshift=70,
        text=f"Limit: {limit:.2f}",
        showarrow=False,
        font=dict(size=10, color="Red"),
        bgcolor="white",
        bordercolor="Red",
        borderwidth=0.5,
        borderpad=4
    )
    

    # Adjusting layout
    fig.update_layout(
        title={
            'text': f"{distribution} distribution",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="n",
        yaxis_title="f(n)",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        ),
        hovermode="x",
        plot_bgcolor='white',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='rgba(230, 230, 230, 0.5)',
        yaxis_gridcolor='rgba(230, 230, 230, 0.5)'
    )

    fig.show()

    #fig.write_image(f"/path/plot_{distribution}.png", scale=3.0)


def evaluation_and_plotting(n_max = 1000, maxiter=1, distribution='uniform' , **parameters):
    """
    Evaluate error values and plot them for various distributions.

    Parameters:
    - n_max (int): Maximum value of n for which to compute the error values. Default is 1000.
    - maxiter (int): Number of iterations for each value of n. Default is 1.
    - distribution (str): Type of distribution ('uniform', 'norm', 'exp', 'beta', 'betaprime'). Default is 'uniform'.
    - **parameters: Distribution-specific parameters. 

    Returns:
    None. Evaluates error values and displays the corresponding plot.
    """
    error_values, limit = evaluation(n_max, maxiter, distribution , **parameters)
    plotting(error_values, limit, distribution)

evaluation_and_plotting(distribution='uniform' , a=2, b=5)

