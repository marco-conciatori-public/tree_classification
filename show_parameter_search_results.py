import global_constants
from import_args import args
from visualization import visualization_utils
from visualization.grid_search_results import top_configurations, graph_2d, graph_3d, loss_per_parameter


# import parameters
parameters = args.import_and_check(global_constants.CONFIG_PARAMETER_PATH)

bar_width = 0.7
top_n = parameters['best_n_configurations']
notebook_execution = parameters['notebook_execution']
excluded_key_list = []
file_number = int(input('Insert file number: '))
parameter_evaluation = visualization_utils.load_evaluation(file_number)
print('\nINFO:')
for key in parameter_evaluation:
    print(f"{key.replace('_', ' ')}: {parameter_evaluation[key]}")

results = parameter_evaluation['results']
print('len(results):', len(results))
print()

search_space = parameter_evaluation['search_space']
parameters_to_plot = visualization_utils.identify_tested_hp(
    search_space=search_space,
    excluded_key_list=excluded_key_list,
)
# print('Number of parameters tested:', len(parameters_to_plot))
print('parameters tested:\n', parameters_to_plot)

excluded_key_list.extend(parameters_to_plot)
if len(parameters_to_plot) == 1:
    graph_2d.plot_parameters(
        hp_to_plot=parameters_to_plot,
        excluded_key_list=excluded_key_list,
        hp_evaluation=parameter_evaluation,
        rotate_x_labels=False,
        save_img=notebook_execution,
    )

elif len(parameters_to_plot) == 2:
    graph_3d.plot_parameters(
        hp_to_plot=parameters_to_plot,
        excluded_key_list=excluded_key_list,
        hp_evaluation=parameter_evaluation,
        bar_width=bar_width,
        save_img=notebook_execution,
    )

# show only selected parameters
select_parameters = ['test_loss']
select_parameters.extend(visualization_utils.extract_parameter_keys(parameters_to_plot=parameters_to_plot))
top_configurations.print_top_n(
    results=results,
    top_n=top_n,
    select_parameters=select_parameters,
)
print()
average_loss = top_configurations.average_loss_per_parameter(parameter_evaluation=parameter_evaluation)
loss_per_parameter.bar_plot(average_loss=average_loss)
