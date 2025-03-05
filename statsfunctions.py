# statsfunctions.py
import numpy                   as np

def rsquaredmixedlm(modelfit):
    """ 
    Calculates Nakagawa Rsquared for mixed linear models
    with first marginal R^2 and then conditional R^2; we'll usually just use conditional
    """
    var_fixed = np.var(modelfit.fittedvalues)
    random_effects = modelfit.random_effects
    var_random = modelfit.cov_re.iloc[0,0]
    var_resid = modelfit.scale
    var_total = var_fixed + var_random + var_resid
    R2_m = var_fixed / var_total # marginal r^2, prop of var explained by fixed effects
    R2_c = (var_fixed + var_random) / var_total
    return R2_m, R2_c

def remove_offsets(fittedmodel, fromdataframe, whichcolumn='Net_MR', indexing=0):
    """
    remove offsets from individual data. fittedmodel should be a statsmodel mixedlm fit, with
    random effects. Offsets are removed from fromdataframe[whichcolumn], based on random
    effects aligned with fromdataframe['uniqsubs'] indexed by indexing=0.
    """
    offsets = fittedmodel.random_effects.values()
    subjectsinds = (i + indexing for i in range(0, len(offsets)))
    subject_to_offset = {subject: offset for subject, offset in zip(subjectsinds, offsets)}
    return fromdataframe.apply(lambda row: row[whichcolumn] - subject_to_offset[row['uniqsubs'].astype(int)], axis=1)