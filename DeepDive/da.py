import torch
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

def compare_groups(
    model,
    adata,
    covariate,
    groupA=None,
    groupB=None,
    fdr_method="fdr_bh",
    background = None
):
    model.eval()

    ## Checks
    # Covariate
    all_covariates = model.discrete_covariate_names + model.continuous_covariate_names
    if covariate not in all_covariates:
        raise ValueError(f"Covariate must be one of {all_covariates}.")
    else:
        if covariate in model.discrete_covariate_names:
            covar_type = "discrete"
        else:
            covar_type = "continuous"

    if (covar_type == "discrete" and groupA is None) or (
        covar_type == "discrete" and groupB is None
    ):
        raise ValueError(
            f"If using a discrete covariate, please set both groupA and groupB."
        )
    if groupA is not None:
        if groupA == groupB:
            raise ValueError(f"roupA and groupB must be different.")

    ## GroupA
    if groupA is not None:
        covariate_levels = list(model.data_register["covars_dict"][covariate].keys())[
            :-1
        ]
        if groupA not in covariate_levels:
            raise ValueError(f"GroupA must be one of {covariate_levels}.")

    ## GroupB
    if groupB is not None:
        groupB_covariate_levels = list(set(covariate_levels) - set([groupA]))
        if groupB not in groupB_covariate_levels:
            raise ValueError(f"GroupB must be one of {groupB_covariate_levels}.")
            

    if covar_type == "discrete":
        embedding = model.known.discrete_covariates_embeddings[covariate]
        embedding_idx_A = torch.tensor(
            model.known.data_register["covars_dict"][covariate][groupA], device=model.device
        )
        embedding_idx_B = torch.tensor(
            model.known.data_register["covars_dict"][covariate][groupB], device=model.device
        )
        embedding_A = embedding(embedding_idx_A)
        embedding_B = embedding(embedding_idx_B)
    else:
        embedding = model.known.continuous_covariates_embeddings[covariate]
        embedding_idx = torch.tensor(0, device=model.device)
        embedding_A = embedding(embedding_idx) 
        embedding_B = torch.zeros_like(embedding_A)
        
        
    base = torch.zeros_like(embedding_A)
    if background is not None:
        
        for key, value in background.items():
            if key in model.discrete_covariate_names:
                embedding = model.known.discrete_covariates_embeddings[key]
                embedding_idx = torch.tensor(
                    model.known.data_register["covars_dict"][key][value], device=model.device
                )
                
                base += embedding(embedding_idx)
                
            else:
                embedding = model.known.continuous_covariates_embeddings[key]
                embedding_idx = torch.tensor(0, device=model.device)
                base += embedding(embedding_idx) * value
                
            
    # GroupA
    groupA_recon = (
        torch.stack(
            [model.known.decoder_list[x](base + embedding_A) for x in range(model.n_decoders)]
        )
        .detach()
        .cpu()
        .numpy() 
    ) 
    # GroupB
    groupB_recon = (
        torch.stack(
            [model.known.decoder_list[x](base + embedding_B) for x in range(model.n_decoders)]
        )
        .detach()
        .cpu()
        .numpy()
    ) 

    ## Process
    # Stats per decoder
    diff, Z, P = [], [], []
    for decoder in range(model.n_decoders):
        diff_tmp = groupA_recon[decoder, :] - groupB_recon[decoder, :]
        Z_tmp = (diff_tmp - diff_tmp.mean()) / diff_tmp.std()
        P_tmp = 2 * (1 - norm.cdf(np.abs(Z_tmp)))
        diff.append(diff_tmp)
        Z.append(Z_tmp)
        P.append(P_tmp)

    # Stacks
    P = np.stack(P, axis=0)
    Z = np.stack(Z, axis=0)
    diff = np.stack(diff, axis=0)

    # Summary
    chi_squared_stat = -2 * np.sum(np.log(P), axis=0)
    df = 2 * model.n_decoders
    P_combined = 1 - chi2.cdf(chi_squared_stat, df)
    Z_combined = np.mean(Z, axis=0)
    diff_combined = np.mean(diff, axis=0)
    reject, pvals_corrected, _, _ = multipletests(
        P_combined, alpha=0.05, method=fdr_method
    )

    # Setup results
    test_res = pd.DataFrame(
        {
            "Feature": adata.var_names,
            "groupA": groupA_recon.mean(axis=0),
            "groupB": groupB_recon.mean(axis=0),
            "Difference": diff_combined,
            "Zscore": Z_combined,
            "Pvalue": P_combined,
            "FDR": pvals_corrected,
        }
    )

    return test_res
