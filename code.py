import numpy as np
import cvxpy as cp
import time


time_sdr_ave,time_admm_ave = [],[] #store the results
value_sdr_ave,value_admm_ave = [], []
n,tau,ms = 50,1,[20,40,80,120,160,200]    #problem settings
for m in ms:
    print(m)
    time_sdr,time_admm = [],[]
    value_sdr,value_admm = [], []
    for _ in range(50):
        HR,HI = np.random.randn(n,m),np.random.randn(n,m) #initialize
        A,B = np.hstack((HR.T,HI.T)),np.hstack((-HI.T,HR.T))
        C_i = np.zeros((m,2*n,2*n))  #record the P_i in constraints
        for i in range(m):
            ci,ci2 = A[i,:].reshape((1,2*n)),B[i,:].reshape((1,2*n))
            C_i[i,:,:] = np.matmul(ci.T,ci) + np.matmul(ci2.T,ci2)

        #using SDR :
        X = cp.Variable((2*n,2*n),symmetric=True)
        constraints = [X >> 0]
        for i in range(m):
            constraints += [cp.trace(C_i[i,:,:]@X) >=tau]
        prob = cp.Problem(cp.Minimize(cp.trace(np.eye(2*n) @ X)),constraints)
        time1 = time.time()
        prob.solve()
        #randomization, L=50
        min_ = 10086  # the optimal value, the initial one should be large enough
        for l in range(50):
            sample_x = np.random.multivariate_normal(mean=np.zeros(2*n),cov=X.value).reshape((2*n,1))
            min_value = 10086
            for i in range(m):
                to_compare = np.matmul(sample_x.T,np.matmul(C_i[i,:,:],sample_x))
                if to_compare < min_value:
                    min_value = to_compare
            sample_x = sample_x / np.sqrt(min_value) #this step guarantee that sample_x is feasible
            to_compare = np.matmul(sample_x.T,sample_x)
            if to_compare < min_:
                min_ = to_compare.item()
        # record the time and optimal value
        time_sdr.append(time.time()-time1)
        value_sdr.append(min_)

        #applying consensus-ADMM
        time2 = time.time()
        H = HR + HI * 1j
        w = np.random.randn(n,1) + np.random.randn(n,1)*1j #initialize
        w = w / np.min(np.absolute(np.matmul(H.T.conjugate(),w))) #scale w to make it a feasible point
        zs = m*w
        us = np.zeros((n,1),dtype=complex)
        rou = 2 * np.sqrt(m)
        eps = 0.0005   #the criterion for stopping the iteration (successive difference)
        alpha= np.zeros((m,1),dtype=complex)
        v = np.zeros((m,1),dtype=complex)
        w0 = np.ones((n,1),dtype=complex)
        iter = 0

        #begin concensus-ADMM
        while True:
            w=  (zs + us)/(m+1/rou)
            if np.sum(np.absolute(w-w0)) <= eps:
                break
            yt = np.matmul(H.T.conjugate(),w)
            diff = np.sqrt(tau)-np.absolute(yt-alpha) #mx1
            v = (diff[:,0] * (yt-alpha)[:,0] / (np.absolute(yt-alpha)[:,0] * np.sum(np.absolute(H)**2,axis=0))).reshape((m,1))
            v[diff < 0] = 0
            zs = m*w -us + np.matmul(H,v)
            us = us + zs - m *w
            alpha = (diff[:,0] * (yt-alpha)[:,0] / (np.absolute(yt-alpha)[:,0])).reshape((m,1))
            alpha[diff<0] = 0
            w0 = w
            iter += 1
        time_admm.append(time.time()-time2)
        value_admm.append(np.sum(np.absolute(w)**2))
    time_sdr_ave.append(np.mean(time_sdr))
    time_admm_ave.append(np.mean(time_admm))
    value_admm_ave.append(np.mean(value_admm))
    value_sdr_ave.append(np.mean(value_sdr))


#begin plotting
import matplotlib.pyplot as plt

plt.plot(ms,value_sdr_ave,'.-',markersize=9)
plt.plot(ms,value_admm_ave,'*-',markersize=9)
plt.xlabel('Number of Users')
plt.ylabel('Optimal value')
plt.legend(['SDR with randomization','Concensus-ADMM'])


plt.plot(ms,time_sdr_ave,'.-',markersize=9)
plt.plot(ms,time_admm_ave,'*-',markersize=9)
plt.xlabel('Number of Users')
plt.ylabel('Computation Time')
plt.legend(['SDR with randomization','Concensus-ADMM'])
