### Risk Premia Estimator 
# Paper: Giglio, Xiu, and Zhang (2021)

# Four-split
# Two-pass
# Lasso 
# Ridge
# rpPCA
# PLS
# 3-pass PCA
# SPCA

import numpy as np
from scipy.linalg import sqrtm, block_diag
from scipy.sparse.linalg import svds
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

def kfoldcv_tsr2(M, ffold, param, tuningrange):
    """
    This function is used to select the best tuning parameters based on time series R2.

    Input:
    M: repeat K-fold CV M times
    ffold: K-fold
    param: parameters for func1
    tuningrange: range of tuning parameters
    """
    if 'pmax' not in param:
        param['pmax'] = 1

    gt = param['gt']
    rt = param['rt']
    pmax = param['pmax']

    T = rt.shape[1]
    tsr2 = np.zeros((ffold, M, len(tuningrange), pmax))
    tr = len(tuningrange)

    for m in range(M):
        kf = KFold(n_splits=ffold)
        for i, (train_index, test_index) in enumerate(kf.split(np.arange(T)-1)):
            rt_test = rt[:, test_index]
            rt_train = rt[:, train_index]
            muhat = np.mean(rt_test, axis=1)
            rhatbar = rt_test - muhat[:, None]
            gt_train = gt[:, train_index]
            gt_test = gt[:, test_index]
            ghatbar = gt_test - np.mean(gt_test, axis=1)[:, None] 
            for jj in range(tr):
                prm = param.copy()
                prm['rt'] = rt_train
                prm['gt'] = gt_train
                prm['tuning'] = tuningrange[jj]
                prm['usep'] = np.arange(1, pmax+1)
                res = SPCA_cv(prm, False)
                mimi = res['mimi']
                for p in range(pmax):
                    tsr2[i, m, jj, p] = np.mean(1 - np.sum((ghatbar.T -\
                        rhatbar.T @ mimi[:, :, p].T)**2)/np.sum((ghatbar.T)**2))
                    
    re = {}
    re['tsr2'] = np.mean(tsr2, axis=(0, 1))
    i1, i2 = np.unravel_index(np.argmax(re['tsr2']), re['tsr2'].shape)
    re['tuning'] = tuningrange[i1]
    re['phat'] = i2+1 
    prm = param.copy()
    prm['usep'] = re['phat']
    prm['tuning'] = re['tuning']
    res = SPCA_cv(prm, False)
    res['pmax'] = re['phat']
    res['tuning'] = prm['tuning']
    res['tsr2'] = re['tsr2']
    return res
       
def three_pass_pca(rt, gt, p, q):
    """
    Perform three-pass PCA.
    This function performs PCA estimates of risk premium (three-pass procedure)

    INPUT
    rt          is n by T matrix
    gt          is d by T factor proxies
    p           is the number of latent factors
    q           is # of lags used in Newy-West standard errors

    OUTPUT
    Gammahat_nozero    is d by 1 vector matrix of risk premia estimates
    eta                is 1 by p vector of estimates
    gamma              is p by 1 vector of esimtates
    avarhat_nozero     is d by 1 vector of the avar of risk premia estimates
    vhat               is p by T vector of factor estimates
    sdf                is 1 by T vector of SDF estimates
    b                  is 1 by N vector of SDF loading
    r2ts               is d by p matrix of time series Rsquare
    r2xs               is 1 by p matrix of cross-sectional Rsquare
    """
    # INITIALIZATION
    T = rt.shape[1]
    n = rt.shape[0]
    d = gt.shape[0]

    # ESTIMATION
    rtbar = rt - np.mean(rt, axis=1, keepdims=True)
    rbar = np.mean(rt, axis=1, keepdims=True)
    gtbar = gt - np.mean(gt, axis=1, keepdims=True)

    # PCA
    U, S, V = np.linalg.svd(rtbar, full_matrices=True)
    S = S.reshape(-1,1)
    gammahat = U[:, :p].T @ rbar / np.diag(S[:p, :p])
    
    # Spanning loadings (eta) and mes. error (what)
    etahat = gtbar @ V[:, :p]
    vhat = V[:, :p].T
    Sigmavhat = vhat @ vhat.T / T
    what = gtbar - gtbar @ V[:, :p] @ V[:, :p].T
    phat = p
    
    # Newey-West Estimation
    Pi11hat = np.zeros((d * phat, d * phat))
    Pi12hat = np.zeros((d * phat, phat))
    Pi22hat = np.zeros((phat, phat))

    for t in range(T):
        Pi11hat += np.outer(np.ravel(what[:, t] @ vhat[:, t].T), np.ravel(what[:, t] @ vhat[:, t].T)) / T
        Pi12hat += np.outer(np.ravel(what[:, t] @ vhat[:, t].T) , vhat[:, t]) / T
        Pi22hat += vhat[:, t] @ vhat[:, t].T / T

        for s in range(min(t, q)):
            Pi11hat += 1 / T * (1 - s / (q + 1)) * (
                np.outer(np.ravel(what[:, t] @ vhat[:, t].T), np.ravel(what[:, t - s] @ vhat[:, t - s].T)) +
                np.outer(np.ravel(what[:, t - s] @ vhat[:, t - s].T), np.ravel(what[:, t] @ vhat[:, t].T))
            )
            Pi12hat += 1 / T * (1 - s / (q + 1)) * (
                np.outer(np.ravel(what[:, t] @ vhat[:, t].T), vhat[:, t - s].T) +
                np.outer(np.ravel(what[:, t - s] @ vhat[:, t - s].T), vhat[:, t].T)
            )
            Pi22hat += 1 / T * (1 - s / (q + 1)) * (
                vhat[:, t] @ vhat[:, t - s].T + vhat[:, t - s] @ vhat[:, t].T
            )

    avarhat_nozero = np.diag(
        np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi11hat @ np.kron(np.linalg.inv(Sigmavhat) @ gammahat, np.eye(d)) / T + \
            np.kron(gammahat.T @ np.linalg.inv(Sigmavhat), np.eye(d)) @ Pi12hat @ etahat.T / T + \
                    etahat @ Pi22hat @ etahat.T / T
    )

    # Estimation of risk premium 
    Gammahat_nozero = etahat @ gammahat
    # Estimation of loadings
    sdf =  1-gammahat.T  @ vhat * T
    B = U[:, :p].T
    fhat =  B @ rt
    fhatbar = fhat - np.mean(fhat, axis=1, keepdims=True)
    b = np.mean(fhat, axis=1, keepdims=True).T @  np.linalg.pinv(fhatbar @ fhatbar.T / T) @ B
    
    # OUTPUT
    res = {
        "Gammahat_nozero": np.ravel(Gammahat_nozero),
        "eta": etahat/T**0.5,
        "gamma": etahat*T**0.5,
        "avarhat_nozero": avarhat_nozero,
        "vhat": T**0.5 * vhat,
        "sdf": np.ravel(sdf),
        "b": b
    }
    return res

def three_pass_pls(rt, gt, p):
    """
    Perform three-pass Partial Least Squares (PLS).
    This function returns the estimator of SDF loading and SDF by PLS given
    the number of factor p.

    INPUT
    rt          is n by T matrix
    gt          is d by T factor proxies
    p           is number of PLS factors

    OUTPUT
    Gammahat_nozero    is d by 1 vector matrix of risk premia estimates
    eta                is 1 by p vector of estimates
    gamma              is p by 1 vector of esimtates
    avarhat_nozero     is d by 1 vector of the avar of risk premia estimates
    vhat               is p by T vector of factor estimates
    sdf                is 1 by T vector of SDF estimates
    b                  is 1 by N vector of SDF loading
    B                  is N by p vector of pls loadings
    """
    # INITIALIZATION
    T = rt.shape[1]
    
    # ESTIMATION
    rtbar = rt - np.mean(rt, axis=1, keepdims=True)
    gtbar = gt - np.mean(gt, axis=1, keepdims=True)
    stats = PLSRegression(n_components=p).fit(rtbar.T, gtbar.T)
    B = stats.x_weights_.T
    
    vhat = B @ rt
    vhatbar = vhat - np.mean(vhat, axis=1, keepdims=True)
    gammahat = np.mean(vhat, axis=1, keepdims=True)
    etahat = gtbar @ vhatbar.T @ np.linalg.pinv(vhatbar @ vhatbar.T) 
    Sigmav = vhatbar @ vhatbar.T / T
    b = gammahat.T @ np.linalg.pinv(Sigmav) @ B
    Gammahat_nozero = etahat @ gammahat
    
    sdf = 1 - np.dot(b, rtbar)
    
    # OUTPUT
    res = {
        "Gammahat_nozero": np.ravel(Gammahat_nozero),
        "eta": etahat.T @ sqrtm(Sigmav),
        "gamma": -1*sqrtm(Sigmav) @ gammahat, 
        "vhat": vhat,
        "b": b,
        "sdf": np.ravel(sdf),
        "B": B
    }
    return res

def Ridge_rp(rt, v_r):
    """
    Perform Ridge regression for risk premia.
    This function returns the estimator of SDF loading and SDF by Ridge regression with
    tuning parameter lambda.

    INPUT
    rt: N by T matrix
    lambda: is parameter for Ridge regression (J by 1 vector)

    OUTPUT
    sdf: Ridge SDF estimator for each lambda (J by T matrix)
    b: estimator of SDF loading for each lambda (N by J matrix)
    """

    # INITIALIZATION
    T = rt.shape[1]
    n = rt.shape[0]
    J = len(v_r)
    b = np.zeros((n, J))

    # ESTIMATION
    mrt = np.mean(rt, axis=1, keepdims=True)
    rtbar = rt - np.tile(mrt, (1,T))
    Sigmahat = rtbar @ rtbar.T / T

    for j in range(J):
        b[:, j] = np.ravel(np.linalg.solve(Sigmahat + v_r[j] * np.eye(n), mrt))

    sdf = 1 - b.T @ rtbar

    # OUTPUT
    res = {
        "sdf": sdf,
        "b": b
    }
    return res

def Lasso_rp(rt, v_l):
    """
    Perform Lasso regression for risk premia.
    This function returns the estimator of SDF loading and SDF by Lasso regression with
    tuning parameter lambda.

    INPUT
    rt: returns (N by T matrix)
    lambda: parameter for Ridge regression (J by 1 vector)

    OUTPUT
    sdf: Ridge SDF estimator for each lambda (J by T matrix)
    b: estimator of SDF loading for each lambda (N by J matrix)
    """

    # INITIALIZATION
    T = rt.shape[1]
    n = rt.shape[0]
    J = len(v_l)
    b = np.zeros((n, J))

    # ESTIMATION
    mrt_o = np.mean(rt, axis=1, keepdims=True)
    c = np.linalg.norm(rt - mrt_o, ord=2) / (T ** 0.5 * n ** 0.5)
    rt = rt / c
    v_l = v_l / c
    mrt = np.mean(rt, axis=1).reshape(-1,1)
    rtbar = rt - np.tile(mrt, (1,T))
    Sigmahat = rtbar @ rtbar.T / T

    max_iter = 3000
    gamma = 1 / np.linalg.norm(Sigmahat)  # stepsize

    for j in range(J):
        w = np.zeros(n)
        v = w.copy()
        for t in range(max_iter-1):
            v_old = v.copy()
            w_prev = w.copy()
            w = v - gamma * (Sigmahat @ v - np.ravel(mrt))
            w = np.sign(w) * np.maximum(np.abs(w) - v_l[j] * gamma, 0)
            v = w + t / (t + 3) * (w - w_prev)
            if np.sum(np.power(v - v_old, 2)) < (np.sum(np.power(v_old, 2)) * 1e-5) or \
                np.sum(np.abs(v - v_old)) == 0:
                break
        b[:, j] = v
        
    sdf = 1 - b.T @ rtbar

    # OUTPUT
    res = {
        "sdf": sdf,
        "b": b / c
    }
    return res

def Best_likelihood(rt, b, Sigma, mu):
    """
    Select the tuning parameter with the largest likelihood.
    This function is used to select the best SDF estimator of Ridge and Lasso
    based on maximum likelihood given true covariance matrix, Sigma and mean, mu

    INPUT:
    rt: n by T matrix
    b: estimator of SDF loading for each lambda (N by J matrix)
    Sigma: true covariance matrix (N by N)
    mu: true mean vector (N by 1)

    OUTPUT:
    b_op: N by 1 vector of best estimates of SDF loading
    sdf_op: 1 by T vector of best SDF estimator
    sdf_all: J by T vector of SDF estimators given J SDF loadings
    """
    # INITIALIZATION
    T = rt.shape[1]
    J = b.shape[1]
    r2 = np.zeros((J,1))

    # ESTIMATION
    mrt = np.mean(rt, axis=1, keepdims=True)
    rtbar = rt - np.tile(mrt, (1,T))

    for j in range(J):
        r2[j] = -(mu - np.dot(Sigma, b[:, j].reshape(-1,1))).T @\
            (np.linalg.solve(Sigma, (mu - np.dot(Sigma, b[:, j].reshape(-1,1)))))

    argmax = np.argmax(r2)
    sdf_all = 1 - np.dot(b.T, rtbar)
    sdf_op = sdf_all[argmax, :]

    # OUTPUT
    res = {'b_op': b[:, argmax], 'b_all': b, 
          'sdf_op': sdf_op, 'sdf_all': sdf_all, 
          'op': argmax, 'r2': r2}
    return res

def risk_premium(gt, sdf_op):
    """
    Calculate the risk premium.
    This function is used to calculate the estimator of risk premium given
    SDF estimates and proxy factor gt

    INPUT:
    gt: factor proxies (d by T)
    sdf_op: the estimator of SDF (1 by T)

    OUTPUT:
    rp: d by 1 vector of risk premium
    """
    T = gt.shape[1]
    gtbar = gt - np.mean(gt, axis=1, keepdims=True)
    mtbar = sdf_op - np.mean(sdf_op)
    res = -gtbar @ mtbar.T / T
    return res

def FM(rt, gt):
    """
    Perform Fama-MacBeth regression.
    This function performs FM estimates of risk premium (two-pass procedure)

    INPUT:
    rt: n by T matrix
    gt: d by T matrix

    OUTPUT:
    Gammahat_nozero: risk premium
    """
    mrt = np.mean(rt, axis=1, keepdims=True)
    rtbar = rt - mrt
    mrt = np.mean(rt, axis=1, keepdims=True)
    gtbar = gt - np.mean(gt, axis=1, keepdims=True)
    betahat = np.linalg.pinv(np.dot(gtbar, gtbar.T)) @ gtbar @ rtbar.T
    gammahat = np.linalg.pinv(np.dot(betahat, betahat.T)) @ betahat @ mrt
    return np.ravel(gammahat)

def rpPCA_cv(param): 
    """
    This function performs rpPCA estimates of risk premium (risk premium PCA)
    """
    # INPUT
    rt = param['rt']
    gt = param['gt']
    pmax = param['pmax']
    tuning = param['tuning']

    # INITIALIZATION
    T = rt.shape[1]
    n = rt.shape[0]
    d = gt.shape[0]
    Gammahat_nozero = np.zeros((d, pmax))
    b = np.zeros((pmax, n))
    mimi = np.zeros((d, n, pmax))

    # ESTIMATION
    gt_mrt = np.mean(gt, axis=1, keepdims=True)
    gtbar = gt - np.tile(gt_mrt, (1,T))
    mu2 = (1+tuning)**0.5 - 1
    R = rt @ (np.eye(T) + mu2/T * np.ones((T, T)))
    U, _, _ = np.linalg.svd(R, full_matrices=True)

    for p in range(pmax):
        vhat = U[:, :p+1].T @ rt
        vhatbar = vhat - np.mean(vhat, axis=1, keepdims=True)
        gammahat = np.mean(vhat, axis=1)
        etahat = gtbar @ vhatbar.T @ np.linalg.inv(vhatbar @ vhatbar.T)
        Gammahat_nozero[:, p] = etahat @ gammahat
        Sigmav = vhatbar @ vhatbar.T / T
        b[p, :] = gammahat.T @ np.linalg.inv(Sigmav) @ U[:, :p+1].T
        mimi[:, :, p] = gtbar @ vhatbar.T @ np.linalg.inv(vhatbar @ vhatbar.T) @ U[:, :p+1].T
        
    mrt = np.mean(rt, axis=1, keepdims=True)
    sdf = 1 - b @ mrt

    # OUTPUT
    res = {'Gammahat_nozero': Gammahat_nozero,
           'b': b,'mimi': mimi,'sdf': sdf}
    return res

def four_split(rt, ft, p2, A1):
    """
    This function performs Four-Split estimates of risk premium.

    INPUT:
    rt: n x T matrix
    ft: pk x T matrix of factors
    p2: number of potential missing factors
    A1: p2 x pk matrix used to create instruments

    OUTPUT:
    Gammahat: pk x 1 vector of risk premia estimator
    Sigmahat: pk x pk matrix of covariance estimator
    """

    T = rt.shape[1]
    n = rt.shape[0]
    pk = ft.shape[0]
    p = pk + p2

    ftbar = ft - np.mean(ft, axis=1, keepdims=True)
    rtbar = rt - np.mean(rt, axis=1, keepdims=True)
    T1 = int(T / 4)
    betahat = np.zeros((n, pk, 4))

    for j in range(4):
        a1 = T1 * j+1
        a2 = T1 * (j+1)
        ftbar1 = ftbar[:, a1:a2]
        rtbar1 = rtbar[:, a1:a2]
        betahat[:, :, j] = rtbar1 @ ftbar1.T @ np.linalg.pinv(ftbar1 @ ftbar1.T)

    mrt = np.mean(rt, axis=1)
    period = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])-1
    res1 = np.zeros((p, 4))

    X = np.zeros((n, p, 4))
    Z = np.zeros((n, 2 * pk, 4))
    subG = np.zeros((p, p, 4))
    tildeZ = np.zeros((p, n, 4))
    resid = np.zeros((n, 1, 4))
    Ze = np.zeros((p, n, 4))

    for j in range(4):
        J = period[j, :]
        X[:, :, j] = np.hstack((betahat[:, :, J[0]], np.vstack((betahat[:, :, J[0]] - betahat[:, :, J[1]]) @ A1)))
        Z[:, :, j] = np.hstack((betahat[:, :, J[2]], np.vstack(betahat[:, :, J[2]] - betahat[:, :, J[3]])))
        Xhat = Z[:, :, j] @ np.linalg.solve(Z[:, :, j].T @ Z[:, :, j], Z[:, :, j].T) @ X[:, :, j]
        tildeZ[:, :, j] = Xhat.T
        subG[:, :, j] = X[:, :, j].T @ Xhat / n
        res1[:, j] = np.linalg.solve(Xhat.T @ Xhat, Xhat.T @ mrt)
        resid[:, :, j] = np.vstack(mrt - X[:, :, j] @ res1[:, j])
        Ze[:, :, j] = tildeZ[:, :, j] * resid[:, :, j].T

    R = np.kron(np.ones((4, 1)), np.vstack((0.25 * np.eye(pk), np.zeros((p2, pk)))))
    G = block_diag(subG[:, :, 0], subG[:, :, 1], subG[:, :, 2], subG[:, :, 3])
    Ze_all = np.vstack((Ze[:, :, 0], Ze[:, :, 1], Ze[:, :, 2], Ze[:, :, 3]))
    Sigma0 = Ze_all @ Ze_all.T / n
    Sigmaf = ftbar @ ftbar.T / T
    Sigmahat = (R.T @ np.linalg.solve(G, Sigma0 @ np.linalg.solve(G.T, R)) / n) + Sigmaf / T
    return {'Gammahat': np.mean(res1[:pk, :], axis=1), 'Sigmahat': Sigmahat}
