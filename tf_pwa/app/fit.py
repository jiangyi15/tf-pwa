from tf_pwa.main import regist_subcommand
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.experimental import extra_amp, extra_data


def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


@regist_subcommand(name="fit")
def fit(config="config.yml", init_params="init_params.json", method="BFGS"):
    """
    simple fit script 
    """
    # load config.yml
    config = ConfigLoader(config)
    
    # set initial parameters if have
    try:
        config.set_params(init_params)
        print("using {}".format(init_params))
    except Exception as e:
        if str(e) != "[Errno 2] No such file or directory: 'init_params.json'":
            print(e)
        print("\nusing RANDOM parameters", flush=True)

    # print("\n########### initial parameters")
    # json_print(config.get_params())

    # fit
    data, phsp, bg, inmc = config.get_all_data()
    try:
        fit_result = config.fit(batch=65000, method=method)
    except KeyboardInterrupt:
        config.save_params("break_params.json")
        raise
    except Exception as e:
        print(e)
        config.save_params("break_params.json")
        raise
    json_print(fit_result.params)
    fit_result.save_as("final_params.json")

    # calculate parameters error
    fit_error = config.get_params_error(fit_result, batch=13000)
    fit_result.set_error(fit_error)
    fit_result.save_as("final_params.json")
    pprint(fit_error)

    print("\n########## fit results:")
    for k, v in config.get_params().items():
        print(k, error_print(v, fit_error.get(k, None)))

    # plot partial wave distribution
    config.plot_partial_wave(fit_result, plot_pull=True)

    # calculate fit fractions
    phsp_noeff = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions({}, phsp_noeff)

    print("########## fit fractions")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)
        else:
            name = i
        fit_frac_string += "{} {}\n".format(name, error_print(fit_frac[i], err_frac.get(i, None)))
    print(fit_frac_string)
