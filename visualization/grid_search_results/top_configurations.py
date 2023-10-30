from visualization import visualization_utils


def print_top_n(results: list, top_n: int = 10, select_parameters: list = None):
    # select_parameters is a dict of lists:
    # {
    #   parameter_1: [value_1, value_2, ...],
    #   parameter_2: [value_1, value_2, ...],
    #   ...
    #   parameter_n: [value_1, value_2, ...]
    #  }
    assert top_n > 0, f'top_n must be > 0, got {top_n}.'

    sorted_results = sorted(results, key=lambda item: item['test_loss'])
    max_len = len(results)
    top_n = min(top_n, max_len)

    for index in range(top_n):
        config = sorted_results[index]
        if select_parameters is not None:
            for parameter in select_parameters:
                print(f'{parameter}: {config[parameter]}', end=', ')
            print()
        else:
            print(config)


def average_loss_per_parameter(parameter_evaluation: dict) -> dict:
    average_loss = {}
    results = parameter_evaluation['results']
    search_space = parameter_evaluation['search_space']
    parameters_to_plot = visualization_utils.identify_tested_hp(search_space=search_space)
    # print('parameters_to_plot:', parameters_to_plot)
    parameter_keys = visualization_utils.extract_parameter_keys(parameters_to_plot=parameters_to_plot)
    # print('parameter_keys:', parameter_keys)
    for parameter_index in range(len(parameters_to_plot)):
        parameter_name = parameter_keys[parameter_index]
        average_loss[parameter_name] = {}
        parameter_space_name = parameters_to_plot[parameter_index]
        parameter_counters = {}
        # print(f'parameter_name: {parameter_name}')
        # print(f'parameter_space_name: {parameter_space_name}')
        for parameter_value in search_space[parameter_space_name]:
            if isinstance(parameter_value, list):
                parameter_value = tuple(parameter_value)
            average_loss[parameter_name][parameter_value] = 0
            parameter_counters[parameter_value] = 0
        # print(f'average_loss[parameter_name]: {average_loss[parameter_name]}')
        # print(f'parameter_counters: {parameter_counters}')

        for config in results:
            parameter_value = config[parameter_name]
            if isinstance(parameter_value, list):
                parameter_value = tuple(parameter_value)
            average_loss[parameter_name][parameter_value] += config['test_loss']
            parameter_counters[parameter_value] += 1
        # print(f'average_loss[parameter_name]: {average_loss[parameter_name]}')
        # print(f'parameter_counters: {parameter_counters}')

        for parameter_value in average_loss[parameter_name]:
            average_loss[parameter_name][parameter_value] /= parameter_counters[parameter_value]

        # print(f'Average {parameter_name} values:')
        # for parameter_value in average_loss[parameter_name]:
        #     print(f'{parameter_value}: {average_loss[parameter_name][parameter_value]}')
        # print()

    return average_loss

