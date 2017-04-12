# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio

class load:
    def __init__(self):
        self.loadData = np.vectorize(self.testFun)


    def getLoadData(self):
        Voltage=np.random.uniform(size=(1,21*10*8))*0.2+0.9
        PTRN = np.array(self.loadModel(0),dtype=float)
        for element in Voltage.flat:
            PTRN=np.row_stack((PTRN, self.loadModel(element)))
        return PTRN

    def testFun(self,a):

        return 0

    def loadModel(self,V=1, P_b=1, P_0=123, Q_0=12, V_0=12):
        L_P = 0.581 + 0.371 * V - 0.0151 * P_b + 0.037 * V * V + 0.016 * V * P_b - 2.457 * 10 ** (-6) * P_b * P_b
        L_Q = 1.193 + 1.064 * V - 0.0108 * P_b + 1.119 * V * V - 0.045 * V * P_b - 0.441 * 10 ** (-4) * P_b * P_b

       #SMPS_P = P_0
        SMPS_Q = Q_0 * (0.029 * V * V + 0.188 * V + 0.272 + 0.236 * V ** 0.033 + 0.236 * V ** 0.033)

        REC_P = 4.6902 * V * V - 6.7404 * V + 2.3222 - 0.85852 * P_0 + 1.8969 * P_0 * V
        REC_Q = Q_0 * (0.266 * V * V + 0.1641 * V - 0.042 + 0.234 * V * V + 0.234 * V * V)

        CL_P = 0.101 * V + 0.099 * V * V + 0.798
        CL_Q = -0.905 * V + 1.402 * V * V + 0.503

        WL_P = -0.634 * V + 0.268 * V * V + 1.336
        WL_Q = -0.905 * V + 1.402 * V * V + 0.503

        R_P = P_0 * (V / V_0) ** 2
       #R_Q = 0

        CTIM3_P = -0.634 * V + 0.268 * V * V + 1.366
        CTIM3_Q = -2.15 * V + 1.751 * V * V + 1.4

        QTIM_P = 0.424 * V - 0.147 * V * V + 0.724
        QTIM_Q = -2.15 * V + 1.751 * V * V + 1.4

        return np.array([L_P, L_Q,  SMPS_Q, REC_P,   REC_Q,   CL_P,   CL_Q,
                        WL_P, WL_Q, R_P, CTIM3_P, CTIM3_Q, QTIM_P, QTIM_Q], dtype=np.float64).T

data=load()

loadData=data.getLoadData()
print((loadData.max(axis=0)-loadData.min(axis=0)))
loadData=(loadData-loadData.min(axis=0))/(loadData.max(axis=0)-loadData.min(axis=0))

weight=np.random.normal(size=(21*10*8,8))
weight=weight/weight.sum(axis=0)

save_fn = 'loadData.mat'
sio.savemat(save_fn, {'loadData': loadData, 'weight': weight})
