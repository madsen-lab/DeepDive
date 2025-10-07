from sklearn.metrics import r2_score

def r2_metric(true, predicted, discrete):
    r2_mean = 0.0
    k = 0
    unique_covar = discrete.unique(dim=0)
    
    for uc in unique_covar:
        true_sub = true[(discrete == uc).min(1).values]
        predicted_sub = predicted[(discrete == uc).min(1).values]
        k += 1
        true_mean = true_sub.mean(0).detach().cpu().numpy()
        predicted_mean = predicted_sub.mean(0).detach().cpu().numpy()
        r2_mean += r2_score(true_mean, predicted_mean)
    
    return r2_mean / k
        
        