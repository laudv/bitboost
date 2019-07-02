import numpy as np

DEFAULTS = {
    "nthreads": 1,                  # xgb, lgbm, cat
    "random_seed": 0,
    "target": None,

    # generic
    "objective": "",                # binary, reg_l1, reg_l2
    "metric": "",                   # comma separated; rmse, error, logloss
    "l2_reg": 0.0,
    "l1_reg": 0.0,

    "niterations": 200,
    "learning_rate": 0.25,
    "example_fraction": 1.0,
    "feature_fraction": 1.0,
    "max_depth": 6,
    "min_gain": 1e-5,
    "huber_alpha": 0.95,

    "categorical": [[]],
}

DEFAULTS_METHOD = {
    "xgb_tree_method": "hist",      # auto, exact, approx, hist
    "xgb_missing": np.nan,
    "lgb_boosting_type": "gbdt",   # gbdt, goss, [rf, dart]
    "lgb_sample_freq": 1,
    "lgb_efb": False,              # Exclusive Feature Bundling
    "lgb_sparse": False,
    "cat_boosting_type": "Plain",   # Ordered, Plain
    "bit_compr_threshold": 0.5,
    "bit_discr_nbits": 4,
    "bit_max_nbins": 16,
    "bit_binary_grad_bound": 1.25,
    "bit_sample_freq": 1,
}

def method_specific_field(field):
    if   field.startswith("bit_"):  return "bit"
    elif field.startswith("xgb_"):  return "xgb"
    elif field.startswith("lgb_"):  return "lgb"
    elif field.startswith("cat_"):  return "cat"
    else:                           return None

def defaults_for(method):
    f = lambda item: method_specific_field(item[0]) == method
    return filter(f, DEFAULTS_METHOD.items())

def gen_params_list(fields, configs=None, defaults=None):
    assert "method" in fields
    assert "target" in fields

    if configs is None and defaults is None:
        common = gen_params_list(fields, [DEFAULTS], DEFAULTS.items())
        methods = fields["method"]
        if not isinstance(methods, list):
            methods = [methods]
        
        configs = []
        for method in methods:
            confs = gen_params_list(fields, common, defaults_for(method))
            for c in confs: c["method"] = method
            configs += confs

        return configs

    # default fields
    for (field, value) in defaults:
        if field in fields:             value = fields[field]
        if not isinstance(value, list): value = [value]

        old_configs = configs
        configs = []
        for config in old_configs:
            for v in value:
                c = config.copy()
                c[field] = v
                configs.append(c)

    return configs

def params_to_csv_fields(params, extra):
    fields = DEFAULTS.copy()
    fields.update(DEFAULTS_METHOD)
    fields["method"] = "unknown"

    del fields["categorical"] # annoying

    for (field, value) in params.items():
        if field in fields:
            fields[field] = value

    fields["train_time0"]   = ""
    fields["train_time"]    = ""
    fields["metric_train0"] = ""
    fields["metric_train"]  = ""
    fields["metric_val"]    = ""
    fields["metric_test"]   = ""
    fields["fold"]          = ""

    for (field, value) in extra.items():
        fields[field] = value

    fields["hash"] = hash(str(params.items()))

    for (field, value) in fields.items():
        if isinstance(fields[field], float):
            fields[field] = round(fields[field], 5)

    return sorted(list(fields.items()), key=lambda t: t[0])

def params_to_csv_header():
    fields = params_to_csv_fields({}, {})
    return ",".join(map(lambda t: str(t[0]), fields))

def params_to_csv_row(params, extra):
    fields = params_to_csv_fields(params, extra)
    return ",".join(map(lambda t: str(t[1]), fields))



    
