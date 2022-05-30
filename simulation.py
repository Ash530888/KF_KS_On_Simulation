import matplotlib.pyplot as plt
import numpy as np

import inference

def rmse(x, x2):
    tot=0
    for i in range(len(x)-1):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))
  
def plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total):
    
    fig, ax = plt.subplots()

    scatter2=ax.scatter(filteredx,filteredy,label='filtered')
    scatter3=ax.scatter(predictedx,predictedy,label='predicted')
    scatter4=ax.scatter(measuredx,measuredy,label='measured')
    scatter5=ax.scatter(smoothedx,smoothedy,label='smoothed')
    scatter1=ax.scatter(truex,truey,label='true')
    
    leg=ax.legend()
    
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

    
def main():
    N=10000
    
    dt = 0.019936
    x_std_meas = y_std_meas = 0.001
    std_acc = 0.1
    
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
                  dtype=np.double)*std_acc**2

    R = np.diag([x_std_meas**2, y_std_meas**2])

    m0 = np.array([0, 0, 0, 0, 0, 0 ])
    
    V0 = np.diag(np.ones(len(m0))*0.001)

    M = B.shape[0]
    P = Z.shape[0]

    w=np.transpose(np.random.multivariate_normal(np.zeros(M), Q, N))
    v=np.transpose(np.random.multivariate_normal(np.zeros(P), R, N))
    
    measured=np.empty((N,2))
    measured[:]=np.nan
    predicted=np.empty((N,2))
    filtered=np.empty((N,2))
    y = np.zeros(shape=(P, N))

    matrixIndex=0

    pCOVs = []
    fCOVs = []
    pMeans = []
    fMeans = []

    noise=np.random.normal(0, 1000, N)

    x = np.add(m0, w[0,0])
    y[:, 0] = np.add(np.dot(Z, x), v[:,0])

    measured[0,:] = y[:,0] + noise[0]

    
    for i in range(1, N):
        x = np.add(np.dot(B, x), w[:,i])
        
        y[:, i] = np.add(np.dot(Z, x), v[:,i])

        measured[i,:] = y[:,i] + noise[i]
   

    filtered = inference.filterLDS(measured.T, B, Q, m0, V0, Z, R)
    smoothed = inference.smoothLDS(B, filtered["xnn"], filtered["Vnn"], filtered["xnn1"], filtered["Vnn1"], m0, V0)

    #plotPositions(truex, filteredx, predictedx, measuredx, smoothedx, truey, filteredy, predictedy, measuredy, smoothedy, total)
    plotPositions(y[0,:], filtered["xnn"][0,0,: N], filtered["xnn1"][0,0,: N], measured[:,0], smoothed["xnN"][0,0,: N], y[1, :], filtered["xnn"][3,0,: N], filtered["xnn1"][3,0,: N], measured[:,1], smoothed["xnN"][3,0,: N], N)
    
    
    
main()
    
