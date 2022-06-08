import plotly.graph_objects as go
import numpy as np

import inference

def rmse(x, x2):
    tot=0
    for i in range(len(x)-1):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))

def plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total):

    trace_mes = go.Scatter(x=measuredx, y=measuredy,
                           mode="markers",
                           name="measured",
                           showlegend=True,
                           )
    trace_true = go.Scatter(x=truex, y=truey,
                            mode="markers",
                            name="true",
                            showlegend=True,
                            )
    trace_filtered = go.Scatter(x=filteredx,
                                y=filteredy,
                                mode="markers",
                                name="filtered",
                                showlegend=True,
                                )
    trace_smoothed = go.Scatter(x=smoothedx,
                                y=smoothedy,
                                mode="markers",
                                name="smoothed",
                                showlegend=True,
                                )
    fig = go.Figure()
    fig.add_trace(trace_mes)
    fig.add_trace(trace_true)
    fig.add_trace(trace_filtered)
    fig.add_trace(trace_smoothed)
    fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    import pdb; pdb.set_trace()


def plotVelocities(true_vel, measured, filtered, smoothed, dt, N):
    fd1 = []
    fd2 = []

    for i in range(1, N):
        fd1.append( (measured[0, i] - measured[0, i-1]) / dt)
        fd2.append( (measured[1, i] - measured[1, i-1]) / dt)

    
##    trace_fd = go.Scatter(x=fd1, y=fd2,
##                           mode="markers",
##                           name="Finite Different Velocity",
##                           showlegend=True,
##                           )
    trace_sm = go.Scatter(x=smoothed[1, 0, :N], y=smoothed[4, 0, :N],
                            mode="markers",
                            name="Smoothed Velocity",
                            showlegend=True,
                            )
    trace_filt = go.Scatter(x=filtered[1, 0, :N], y=filtered[4, 0, :N],
                            mode="markers",
                            name="Filtered Velocity",
                            showlegend=True,
                            )
    trace_true = go.Scatter(x=true_vel[0, :], y=true_vel[1, :],
                            mode="markers",
                            name="True Velocity",
                            showlegend=True,
                            )
    
    fig = go.Figure()
    fig.add_trace(trace_sm)
##    fig.add_trace(trace_fd)
    fig.add_trace(trace_filt)
    fig.add_trace(trace_true)
    fig.update_layout(xaxis_title="velocity x", yaxis_title="velocity y",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    #import pdb; pdb.set_trace()


def plotAccelerations(true_vel, true_acc, filtered, smoothed, dt, N):
    fd1 = []
    fd2 = []

    for i in range(1, N-1):
        fd1.append( (true_vel[0, i] - true_vel[0, i-1]) / dt)
        fd2.append( (true_vel[1, i] - true_vel[1, i-1]) / dt)

    
##    trace_fd = go.Scatter(x=fd1, y=fd2,
##                           mode="markers",
##                           name="Finite Different Acc.",
##                           showlegend=True,
##                           )
    trace_sm = go.Scatter(x=smoothed[2, 0, :N], y=smoothed[5, 0, :N],
                            mode="markers",
                            name="Smoothed Acc.",
                            showlegend=True,
                            )
    trace_filt = go.Scatter(x=filtered[2, 0, :N], y=filtered[5, 0, :N],
                            mode="markers",
                            name="Filtered Acc.",
                            showlegend=True,
                            )
    trace_true = go.Scatter(x=true_acc[0, :], y=true_acc[1, :],
                            mode="markers",
                            name="True Acc.",
                            showlegend=True,
                            )
    
    fig = go.Figure()
##    fig.add_trace(trace_fd)
    fig.add_trace(trace_filt)
    fig.add_trace(trace_true)
    fig.add_trace(trace_sm)
    fig.update_layout(xaxis_title="Acc. x", yaxis_title="Acc. y",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()


def main():
    N=10000

    # LESS NOISY SIM
##    dt = 0.019936
##    x_std_meas = y_std_meas = 0.0001
##    std_acc = 0.01

    # MORE NOISY SIM
    dt = 1e-5
    x_std_meas = y_std_meas = 1e2
    std_acc = 1e4

    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, .5*dt**2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]],
                 dtype=np.double)

    Z = np.matrix([[1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]],
                  dtype=np.double)

    Q =  np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)*(std_acc**2)

    R = np.diag([x_std_meas**2, y_std_meas**2])

    m0 = np.array([0, 0, 0, 0, 0, 0 ])

    # V0 = np.diag(np.ones(len(m0))*0.001)
    V0 = np.diag(np.ones(len(m0))*(0.001**2))

    M = B.shape[0]
    P = Z.shape[0]

    w = np.transpose(np.random.multivariate_normal(np.zeros(M), Q, N))
    v = np.transpose(np.random.multivariate_normal(np.zeros(P), R, N))

    # measured=np.empty((N,2))
    measured = np.empty((P, N))
    true_vel = np.empty((2, N))
    true_acc = np.empty((2, N))
    measured[:] = np.nan
    x = np.empty((M, N))
    # predicted=np.empty((N,2))
    # filtered=np.empty((N,2))
    # y = np.zeros(shape=(P, N))

    # noise=np.random.normal(0, 1000, N)

    # x = np.add(m0, w[0,0])
    x0 = np.random.multivariate_normal(m0, V0)
    measured[:, 0] = np.add(np.dot(Z, x0).squeeze(), v[:,0])

    # measured[0,:] = y[:,0] + noise[0]

    # for i in range(1, N):
    for i in range(N):
        if i==0:
            x[:, i] = np.add(np.dot(B, x0), w[:,i])
        else:
            x[:, i] = np.add(np.dot(B, x[:, i-1]), w[:,i])

        true_vel[0, i] = x[1, i]
        true_vel[1, i] = x[4, i]

        true_acc[0, i] = x[2, i]
        true_acc[1, i] = x[5, i]
        
        measured[:, i] = np.add(np.dot(Z, x[:, i]).squeeze(), v[:, i])

        # measured[i,:] = y[:,i] + noise[i]

    filtered = inference.filterLDS(measured, B, Q, m0, V0, Z, R)
    smoothed = inference.smoothLDS(B, filtered["xnn"], filtered["Vnn"], filtered["xnn1"], filtered["Vnn1"], m0, V0)

    # plotVelocities(true_vel, measured, filtered, smoothed, dt, N)
    plotVelocities(true_vel, measured, filtered["xnn"], smoothed["xnN"], dt, N)

    # plotAccelerations(true_vel, true_acc, filtered, smoothed, dt, N)
    plotAccelerations(true_vel, true_acc, filtered["xnn"], smoothed["xnN"], dt, N)
    
    # plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total)
    # plotPositions(y[0,:], filtered["xnn"][0,0,: N], filtered["xnn1"][0,0,: N], measured[:,0], smoothed["xnN"][0,0,: N], y[1, :], filtered["xnn"][3,0,: N], filtered["xnn1"][3,0,: N], measured[:,1], smoothed["xnN"][3,0,: N], N)
    plotPositions(x[0, :], filtered["xnn"][0, 0, :], filtered["xnn1"][0, 0, :],
                  measured[0, :], smoothed["xnN"][0, 0, :], x[3, :],
                  filtered["xnn"][3, 0, :], filtered["xnn1"][3, 0, :],
                  measured[1, :], smoothed["xnN"][3, 0, :], N)

    
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
