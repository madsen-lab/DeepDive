def counterfactual_prediction(
    model,
    adata_object,
    covar_group_map,
    covars_to_add=None,
    add_unknown=True,
    library_size="observed",
    batch_size=256,
    num_workers=10,
    use_decoder="all",
):

    model.eval()

    adata = adata_object.copy()

    all_covariates = model.discrete_covariate_names + model.continuous_covariate_names
    for covariate in covar_group_map.keys():
        if covariate not in all_covariates:
            raise ValueError(
                f"The covariate {covariate} is not part of covariates used during training: {all_covariates}."
            )

    for covariate, group in covar_group_map.items():
        adata.obs[covariate + "_org"] = adata.obs[covariate].copy()
        adata.obs[covariate] = group

    recon = model.predict(
        adata,
        covars_to_add=covars_to_add,
        batch_size=batch_size,
        num_workers=num_workers,
        add_unknown=add_unknown,
        use_decoder=use_decoder,
        library_size=library_size,
    )
    return recon
