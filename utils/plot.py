import os
import datetime
import torch
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, colors
import seaborn as sns


from utils.calc import clarke_error_grid, parkes_error_grid, MMOLL_MGDL


sns.set()
sns.set_context('paper')
sns.set_style('white')


def create_gp_training_plots(logs, save_to=None):
    logs = logs.reset_index()
    logs['acc_batch_num'] = np.nan
    batch_per_epoch = {}
    for fold in logs.fold.unique():
        batch_per_epoch[fold] = logs.loc[logs.fold == fold, 'batch'].max()
    for i in range(len(logs)):
        batch = logs.batch.iloc[i]
        epoch = logs.epoch.iloc[i]
        fold = logs.fold.iloc[i]
        bpe = batch_per_epoch[fold]
        logs['acc_batch_num'].iloc[i] = (epoch-1) * bpe + batch

    batch_results = logs[~logs.batch.isna()]

    # Loss over batches (training)
    plt.figure()
    sns.lineplot(data=batch_results, x='acc_batch_num', y='train_loss', hue='fold').set(title='GP Training Batch Loss')
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend(title="fold")
    if save_to:
        # plt.savefig(save_to + '/training_batch_loss.pdf', bbox_inches='tight')
        plt.savefig(save_to + '/training_batch_loss.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # Hyperparameters and learning rate over batches
    cols = batch_results.columns.tolist()
    hps = cols[cols.index("kernel")+1:]
    hps.remove('acc_batch_num')
    for hp in hps:
        plt.figure()
        sns.lineplot(data=batch_results, x='acc_batch_num', y=hp, hue='fold').set(title='GP ' + hp + ' Convergence')
        plt.xlabel("batch")
        plt.ylabel(hp)
        plt.legend(title="fold")
        if save_to:
            # plt.savefig(save_to + '/training_batch_loss.pdf', bbox_inches='tight')
            plt.savefig(save_to + '/' + hp + '.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def create_plots_around_indices(df, indices, dir_name, sensor, file_id, mark_indices=None, mark_indices_2=None, mark_indices_3=None):
    """
    Creates plots of the dataframe at the given indices. Special indices (duplicates, high density regions, PISA) can
    be marked in different colors.
    """
    path = "/local/home/ansass/Thesis/icarus/plots/" + dir_name + "/" + sensor + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    set_marked_indices = set(mark_indices) if mark_indices is not None else set()
    set_marked_indices_2 = set(mark_indices_2) if mark_indices_2 is not None else set()
    set_marked_indices_3 = set(mark_indices_3) if mark_indices_3 is not None else set()
    indices_marked = list(set_marked_indices.union(set_marked_indices_2).union(set_marked_indices_3))

    ts = df["datetime"]
    df_duplicate_ts = list(ts[ts.duplicated()])
    plot_indices = list(indices)
    num_points_plot = 18 if sensor == 'libre' else 50
    num = 1
    while num < len(plot_indices):
        if plot_indices[num] - plot_indices[num - 1] < num_points_plot:
            plot_indices.pop(num)
        else:
            num += 1

    label_blue = 'regular'
    label_red = 'scan'
    label_yellow = ''
    label_green = ''

    if 'scan_regions' in dir_name:
        label_red = 'scan'
    elif 'pisa_regions' in dir_name:
        label_red = 'PISA'
    elif 'high_density_regions' in dir_name:
        label_red = 'high density region'
    elif 'backward_datetime_jump_regions' in dir_name:
        label_red = 'backward jump region'

    # sns.set_theme(style="darkgrid")
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (4, 4)
    ylim = (1, 20)
    tick_interval = 2

    for plot_index in plot_indices:
        min_index, max_index = int(plot_index - num_points_plot/2), int(plot_index + num_points_plot/2)
        min_index = max(df.index[0], min_index)
        max_index = min(df.index[-1], max_index)
        plot_indices_unmarked_region = [index for index in range(min_index, max_index) if index not in indices_marked]
        df_plot_region = df.loc[plot_indices_unmarked_region]

        # Plotting
        fig, ax = plt.subplots(figsize=size)
        ax.scatter(x=df_plot_region["datetime"], y=df_plot_region["glucose_value"], color='tab:blue', s=10, marker='o', label=label_blue)
        # sns.scatterplot(x="datetime", y="glucose_value", color=["blue"], data=df_plot_region, ax=ax, label=label_blue)
        for mark_ind, color, marker, label in zip([mark_indices, mark_indices_2, mark_indices_3], ['tab:red', 'tab:yellow', 'tab:green'], ['x', 'v', 's'], [label_red, label_yellow, label_green]):
            if mark_ind is not None:
                mark_indices_plot = [index for index in mark_ind if min_index <= index <= max_index]
                mark_points_plot = df.loc[mark_indices_plot]
                ax.scatter(x=mark_points_plot["datetime"], y=mark_points_plot["glucose_value"], color=color, s=10, marker=marker, label=label)
        if any(plot_type in dir_name for plot_type in ["duplicates", "high_density_regions", 'duplicate_timestamp_regions']):
            for ts in df_plot_region["datetime"]:
                if ts in df_duplicate_ts:
                    plt.axvline(x=ts, color="tab:red", zorder=0, lw=1, label='duplicate')
                # else:
                #     plt.axvline(x=ts, color="tab:blue", zorder=0, lw=0.5)

        # Formatting
        ax.grid(linewidth=0.4)
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=tick_interval))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
        plt.ylim(bottom=ylim[0], top=ylim[1])
        plt.legend(title='CGM measurements', framealpha=1)
        plt.tight_layout()

        plt.savefig(path + str(file_id) + "_" + df["datetime"].loc[plot_index].strftime("%Y-%-m-%d_%H:%M"), dpi=600)


def create_pisa_plots(df, indices, dir_name, sensor, patient_id):
    """
    Creates plots of the dataframe at the given PISA indices
    :param df: patient_data (columns: datetime, glucose_values)
    :param indices: dataframe containing 4 columns (aggressive, trial, nominal, cautious) with indices indicating the
                    PISA points identified by the corresponding strategy
    :param dir_name: name of directory where to save the plots
    :param sensor: name of the sensor
    :param patient_id: patient's ID
    :return: None
    """
    sns.set_theme(style="darkgrid")
    path = "/local/home/ansass/Thesis/icarus/plots/" + dir_name + "/" + sensor + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    plot_indices = list(indices.aggressive)
    num = 1
    while num < len(plot_indices):
        if plot_indices[num] - plot_indices[num - 1] < 50:
            plot_indices.pop(num)
        else:
            num += 1
    for plot_index in plot_indices:
        min_index, max_index = plot_index - 25, plot_index + 25
        df_plot_region = df.loc[min_index:max_index]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="datetime", y="glucose_value", palette=["blue"], data=df_plot_region, ax=ax)

        aggressive_indices_plot = [index for index in indices.aggressive if min_index <= index <= max_index]
        trial_indices_plot = [index for index in indices.trial if min_index <= index <= max_index]
        nominal_indices_plot = [index for index in indices.nominal if min_index <= index <= max_index]
        cautious_indices_plot = [index for index in indices.cautious if min_index <= index <= max_index]

        aggressive_points_plot = df.loc[aggressive_indices_plot]
        trial_points_plot = df.loc[trial_indices_plot]
        nominal_points_plot = df.loc[nominal_indices_plot]
        cautious_points_plot = df.loc[cautious_indices_plot]

        aggressive_points_plot['glucose_value'] = 2.75
        trial_points_plot['glucose_value'] = 2.0
        nominal_points_plot['glucose_value'] = 1.25
        cautious_points_plot['glucose_value'] = 0.5

        aggressive_points_plot['parameter-set'] = 'aggressive'
        trial_points_plot['parameter-set'] = 'trial'
        nominal_points_plot['parameter-set'] = 'nominal'
        cautious_points_plot['parameter-set'] = 'cautious'

        all_pisa_points_plot = pd.concat([aggressive_points_plot,
                                         trial_points_plot,
                                         nominal_points_plot,
                                         cautious_points_plot], ignore_index=True)
        pisa_timestamps = all_pisa_points_plot.datetime.drop_duplicates()

        # colors = ["#2B7BBA", "#CC4125", "#F6B26B", "#FFD966", "#93C47D"]
        # sns.set_palette(sns.color_palette(colors))
        # ax_2 = sns.scatterplot(x="datetime", y="glucose_value", hue='parameter-set', data=all_pisa_points_plot, ax=ax)
        ax_2 = sns.scatterplot(x="datetime", y="glucose_value", color="#CC4125", data=aggressive_points_plot, ax=ax, label='aggressive')
        ax_3 = sns.scatterplot(x="datetime", y="glucose_value", color="#F6B26B", data=trial_points_plot, ax=ax, label='trial')
        ax_4 = sns.scatterplot(x="datetime", y="glucose_value", color="#FFD966", data=nominal_points_plot, ax=ax, label='nominal')
        ax_5 = sns.scatterplot(x="datetime", y="glucose_value", color="#93C47D", data=cautious_points_plot, ax=ax, label='cautious')

        for ts in pisa_timestamps:
            plt.axvline(x=ts, color="cornflowerblue", zorder=0, lw=0.5)

        ax.set(xlabel='Time', ylabel='Glucose Value [mmol/L]')
        xfmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        plt.gcf().autofmt_xdate()
        plt.ylim(bottom=0)
        plt.ylim(top=20)
        plt.legend(title='parameter-set')
        plt.savefig(path + str(patient_id) + "_" + df["datetime"].loc[plot_index].strftime("%Y-%-m-%d_%H:%M"), dpi=600)


def plot_gaussian_process(inputs, target, inputs_gp, output_gp, style='white', save_to=None):
    """
    Plot the gp posterior for a sample consisting of (inputs, target)
    """
    mean_color = 'tab:blue'
    if style == 'darkgrid':
        sns.set_theme(style="darkgrid")
        mean_color = '#1521FF'
        # std_color =
    lower, upper = output_gp.confidence_region()

    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(24, 6))

    ax.plot(inputs_gp.cpu(), output_gp.mean.detach().cpu(),
            color=mean_color, label=r'prediction $\mu$')
    ax.fill_between(inputs_gp.cpu(), lower.detach().cpu(), upper.detach().cpu(),
                    color=mean_color, alpha=0.2, label=r'prediction $2\sigma$')

    sns.scatterplot(x=inputs.cpu(), y=target.cpu(), s=10, ax=ax,
                    color='black', label='existing data')
    ax.set_xticklabels([(datetime.datetime(2021, 11, 8) + \
                         datetime.timedelta(minutes=i)).strftime('%H:%M') for i in ax.get_xticks()])
    ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
    plt.ylim(bottom=0, top=20)
    plt.legend()
    # plt.ylim(bottom=-5, top=25)
    # plt.legend('', frameon=False)

    if save_to:
        plt.savefig(save_to + '.pdf', bbox_inches='tight')
        plt.savefig(save_to + '.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_mean_glucose(
        x_times,
        y_mean_glucose,
        sigma_std_glucose,
        style='white',
        save_to=None
):
    """
    Plot the mean and std of glucose values over time
    """
    mean_color = 'tab:blue'
    if style == 'darkgrid':
        sns.set_theme(style="darkgrid")
        mean_color = '#1521FF'

    lower, upper = y_mean_glucose - sigma_std_glucose, y_mean_glucose + sigma_std_glucose

    fig, ax = plt.subplots(figsize=(24, 6))

    ax.plot(x_times, y_mean_glucose, color=mean_color, label=r'Average Glucose Measurement')
    ax.fill_between(x_times, lower, upper, color=mean_color, alpha=0.2, label=r'Standard Deviation $2\sigma$')

    ax.set_xticklabels([(datetime.datetime(2021, 11, 8) + datetime.timedelta(minutes=i)).strftime('%H:%M') for i in ax.get_xticks()])
    ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
    plt.ylim(bottom=0, top=20)
    plt.legend()
    # plt.ylim(bottom=-5, top=25)
    # plt.legend('', frameon=False)

    if save_to:
        plt.savefig(save_to + '.pdf', bbox_inches='tight')
        plt.savefig(save_to + '.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()


def plot_interpolation(
        measure_datetimes,
        measure_values,
        input_datetimes,
        inputs,
        target_datetimes,
        targets,
        window_id,
        save_dir
        ):
    """
    Visualises the sample's interpolation
    """
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (14.5, 4)   # (16, 6)
    ylim = (1, 20)

    fig, ax = plt.subplots(figsize=size)

    # Measurements, Inputs & Targets
    ax.scatter(measure_datetimes, measure_values, color='black', marker='x', s=25, label="measurement", zorder=3)
    ax.plot(measure_datetimes, measure_values, color='black', lw=1, label="linear interpolation", zorder=1)
    ax.scatter(input_datetimes, inputs, color='tab:blue', marker='o', s=20, label="input", zorder=2)
    ax.scatter(target_datetimes, targets, color='tab:red', marker='s', s=20, label="target", zorder=2)

    # Formatting
    ax.grid(linewidth=0.4)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.legend(framealpha=1)
    # plt.tight_layout()

    # Saving
    filepath = os.path.join(save_dir, window_id)
    plt.savefig(filepath, bbox_inches='tight', dpi=600)


def plot_forecast(file_id,
                  date,
                  input_datetimes,
                  inputs,
                  target_datetimes,
                  targets,
                  mus,
                  sigmas,
                  measure_datetimes,
                  measure_values,
                  domain_name,
                  kpis=None,
                  save_to=None,
                  plot_show=True
                  ):
    """
    Visualises the model's forecast for the passed sample
    """
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    mean_color = '#1521FF'

    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (8, 4)   # (16, 6)
    ylim = (1, 20)

    fig, ax = plt.subplots(figsize=size)

    # Measurements, Inputs & Targets
    # ax.scatter(measure_datetimes, measure_values, s=15, marker='x', color='black', label="measurement")
    ax.plot(input_datetimes, inputs, color='black', marker='x', markevery=3, ls='-', lw=1, label="input", zorder=1)
    ax.plot(target_datetimes, targets, color='black', ls='--', lw=1, label="target", zorder=3)

    # Predictions
    ax.plot(target_datetimes, mus, color=mean_color, marker='o', markevery=3, ls='-', lw=1, label=r"prediction $\mu$", zorder=4)
    # ax.plot(target_datetimes, mus_1, color='g', marker='v', markevery=3, ls='-', lw=1, label=r"$\mu$ NLL+PEG", zorder=5)
    ax.fill_between(target_datetimes, mus - 1 * sigmas, mus + 1 * sigmas, color=mean_color, alpha=0.2, label=r'prediction $\sigma$', zorder=2)
    ax.fill_between(target_datetimes, mus - 2 * sigmas, mus + 2 * sigmas, color=mean_color, alpha=0.1, label=r'prediction 2$\sigma$', zorder=1)

    # Hypo- & Hyper Thresholds
    ax.axhline(70 / 18.02, color='black', ls='dotted', label='Target Threshold', zorder=0)
    ax.axhline(180 / 18.02, color='black', ls='dotted', zorder=0)

    # Formatting
    ax.grid(linewidth=0.4)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.legend(framealpha=1)

    # kpis_string = ''.join([f"\n{k}: {round(v, 2)}" for k, v in kpis.items()]) if kpis else ''
    # plt.title(f"File {file_id} on {date} ({domain_name})" + kpis_string)
    #
    # ax.scatter(measure_datetimes, measure_values, s=15, marker='x', color='black', label="measurement")
    # ax.plot(input_datetimes, inputs, 'k-', label='input')
    # ax.plot(target_datetimes, targets, color='grey', label='target')
    # ax.plot(target_datetimes, mus, color=mean_color, ls='-', label=r"prediction $\mu$")
    #
    # # different shading for different confidence regions
    # ax.fill_between(target_datetimes, mus - 1 * sigmas, mus + 1 * sigmas, color=mean_color, alpha=0.2, label=r'prediction $\sigma$')
    # ax.fill_between(target_datetimes, mus - 2 * sigmas, mus + 2 * sigmas, color=mean_color, alpha=0.1, label=r'prediction $2\sigma$')
    #
    # myFmt = mdates.DateFormatter('%H:%M')
    # ax.xaxis.set_major_formatter(myFmt)
    # ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
    # plt.ylim(bottom=0, top=20)
    # plt.legend()

    if save_to:
        # plt.savefig(save_to + f'{plot_num}' + '.pdf', bbox_inches='tight')
        plt.savefig(save_to + '.png', bbox_inches='tight', dpi=600)
    if plot_show:
        plt.show()
    plt.close()


def compare_forecasts(input_datetimes,
                      inputs,
                      target_datetimes,
                      targets,
                      mus_0,
                      mus_1,
                      sigmas_0,
                      sigmas_1,
                      model_0_name='NLL',
                      model_1_name='NLL+PEG',
                      save_to=None,
                      plot_show=True
                      ):
    """
    Visualises the models' forecast for the passed samples
    """
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')

    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (14.5, 4)
    ylim = (1, 20)
    fig, ax = plt.subplots(figsize=size)

    # Measurements, Inputs & Targets
    ax.plot(input_datetimes, inputs, color='black', marker='x', markevery=3, ls='-', lw=1, label="input", zorder=1)
    ax.plot(target_datetimes, targets, color='black', ls='--', lw=1, label="target", zorder=3)

    # Predictions
    ax.plot(target_datetimes, mus_0, color='r', marker='o', markevery=3, ls='-', lw=1, label=rf"$\mu$ {model_0_name}", zorder=4)
    ax.plot(target_datetimes, mus_1, color='g', marker='v', markevery=3, ls='-', lw=1, label=rf"$\mu$ {model_1_name}", zorder=5)
    ax.fill_between(target_datetimes, mus_0 - 1 * sigmas_0, mus_0 + 1 * sigmas_0, color='r', alpha=0.2, label=rf'$\sigma$ {model_0_name}', zorder=1)
    ax.fill_between(target_datetimes, mus_1 - 1 * sigmas_1, mus_1 + 1 * sigmas_1, color='g', alpha=0.2, label=rf'$\sigma$ {model_1_name}', zorder=2)

    # Hypo- & Hyper Thresholds
    ax.axhline(70 / 18.02, color='black', ls='dotted', label='Target Threshold', zorder=0)
    ax.axhline(180 / 18.02, color='black', ls='dotted', zorder=0)

    # Formatting
    ax.grid(linewidth=0.4)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.set(xlabel='Time [hh:mm]', ylabel='Glucose Value [mmol/L]')
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.legend(framealpha=1)

    # Saving, Showing & Closing
    if save_to:
        # plt.savefig(save_to + f'{plot_num}' + '.pdf', bbox_inches='tight')
        plt.savefig(save_to + '.png', bbox_inches='tight', dpi=600)
    if plot_show:
        plt.show()
    plt.close()


def format_error_grid(ax, grid_type='clarke', lims=None, units='mmoll'):
    if lims is None:
        lims = [0., 450.]
    if grid_type == 'clarke':
        lines = np.array([[lims, lims],
                         [[lims[0], 70/1.2],     [70, 70]],
                         [[70, 70],              [lims[0], 70*0.8]],
                         [[70, lims[1]],         [70*0.8, lims[1]*0.8]],
                         [[70/1.2, lims[1]/1.2], [70, lims[1]]],
                         [[lims[0], 70],         [180, 180]],
                         [[180, 180],            [lims[0], 70]],
                         [[180, lims[1]],        [70, 70]],
                         [[70, 70],              [70*1.2, lims[1]]],
                         [[240, 240],            [70, 180]],
                         [[240, lims[1]],        [180, 180]],
                         [[182*(5/7), 180],      [lims[0], 70]],
                         [[70, lims[1]],         [180, lims[1]+110]]])
    elif grid_type == 'parkes':
        lines = np.array([[lims, lims],
                          [[lims[0], 30], [50, 50]],    # B upper
                          [[30, 140], [50, 170]],       # B upper
                          [[140, 280], [170, 380]],     # B upper
                          [[280, 430], [380, 550]],     # B upper
                          [[50, 50], [lims[0], 30]],    # B lower
                          [[50, 170], [30, 145]],       # B lower
                          [[170, 385], [145, 300]],     # B lower
                          [[385, 550], [300, 450]],     # B lower

                          [[lims[0], 30], [60, 60]],    # C upper
                          [[30, 50], [60, 80]],         # C upper
                          [[50, 70], [80, 110]],        # C upper
                          [[70, 260], [110, 550]],      # C upper
                          [[120, 120], [lims[0], 30]],  # C lower
                          [[120, 260], [30, 130]],      # C lower
                          [[260, 550], [130, 250]],     # C lower

                          [[lims[0], 25], [100, 100]],  # D upper
                          [[25, 50], [100, 125]],       # D upper
                          [[50, 80], [125, 215]],       # D upper
                          [[80, 125], [215, 550]],      # D upper
                          [[250, 250], [lims[0], 40]],  # D lower
                          [[250, 550], [40, 150]],      # D lower

                          [[lims[0], 35], [150, 155]],  # E upper
                          [[35, 50], [155, 550]]])      # E upper

    s = 1/MMOLL_MGDL if units == 'mmoll' else 1

    lines *= s

    for x, y in lines:
        ax.plot(x, y, c='k', lw=1)

    ax.set_xlim(np.array(lims)*s)
    ax.set_ylim(np.array(lims)*s)
    if grid_type == 'clarke':
        texts = np.array([(30, 15, 'A'),
                          (414, 260, 'B'),
                          (324, 414, 'B'),
                          (160, 414, 'C'),
                          (160, 15, 'C'),
                          (30, 140, 'D'),
                          (414, 120, 'D'),
                          (30, 260, 'E'),
                          (414, 15, 'E'),
                          ])
    elif grid_type == 'parkes':
        texts = np.array([(280, 315, 'A'),
                          (315, 280, 'A'),
                          (215, 355, 'B'),
                          (355, 215, 'B'),
                          (145, 380, 'C'),
                          (380, 125, 'C'),
                          (70, 390, 'D'),
                          (390, 40, 'D'),
                          (20, 392.5, 'E')
                          ])
    texts = pd.DataFrame(texts, columns=['x', 'y', 'text'])
    texts[['x', 'y']] = texts[['x', 'y']].astype(float) * s

    for _, (x, y, text) in texts.iterrows():
        # ax.text(x, y, text, fontsize=15)
        ax.text(x, y, s=text, fontsize=10)


def plot_error_grid(output, target, domain_name='', grid_type='clarke', units='mmoll', save_to=None, plot_show=True):
    if grid_type == 'clarke':
        areas = clarke_error_grid(output, target, units)
    elif grid_type == 'parkes':
        areas = parkes_error_grid(output, target, units)
    areas = pd.DataFrame({c: mask.flatten().cpu().numpy() for c, mask in areas.items()})

    colors = sns.color_palette('RdYlGn_r', len(areas.keys()))
    colors[2] = (246/256, 245/256, 128/256) # (1, 1, 128/256)
    colors = {k: c for k, c in zip(areas.keys(), colors)}

    df = pd.DataFrame(torch.stack([output.flatten().cpu(), target.flatten().cpu()]).T.numpy(), columns=['output', 'target'])
    df['area'] = (areas == True).idxmax(axis=1)

    sns.set()
    sns.set_context('paper')
    sns.set_style('white')

    matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (4, 4)

    # fig, ax = plt.subplots(figsize=(8, 7))
    fig, ax = plt.subplots(figsize=size)
    # note: we cannot use scatterplot in one line because it does not use hue correctly when areas of the CEG are missing
    for k in areas.keys():
        sns.scatterplot(data=df[df['area'] == k], x='target', y='output', color=colors[k], alpha=.8, s=1.5)
    format_error_grid(ax, grid_type=grid_type, units=units)

    ax.set(xlabel='True glucose concentration [mmol/L]', ylabel='Predicted glucose concentration [mmol/L]')
    # plt.legend(framealpha=1)
    # plt.xlabel('True glucose concentration [mmol/L]')
    # plt.ylabel('Predicted glucose concentration [mmol/L]')

    # plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.06), fancybox=False)
    if grid_type == 'clarke':
        name = 'Clarke'
    elif grid_type == 'parkes':
        name = 'Parkes'
    else:
        name = ''
    plt.title(name + f" Error Grid ({domain_name})\n"+'    '.join(["{:s}: {:.2f}%".format(k, areas[k].sum()/len(df)*100) for k in areas.keys()]))

    if save_to:
        # plt.savefig(save_to+'.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(save_to+'.png', bbox_inches='tight', dpi=600)
    if plot_show:
        plt.show()
    plt.close()


def plot_peg_loss(
        X,
        Y,
        L,
        L_y,
        save_dir,
        version='2'
        ):

    contour_points = [(550, 450), (260, 550), (0, 100), (35, 155)]
    contour_levels_loss = [0, 0.1, 0.25, 0.5, 1, 2, 10, 20, 40, 60, 80]
    contour_levels_derivative = [-4, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 4, 5]

    if version == '1':
        R = np.exp(-X / 10) - np.exp(-Y / 10)
        L = abs(R)

        def peg_loss(x, y):
            s = 1 / MMOLL_MGDL
            return abs(np.exp(-x * s / 10) - np.exp(-y * s / 10))
        contour_levels_loss = [0] + [peg_loss(p[0], p[1]) for p in contour_points] + [100]

    # Plot settings
    sns.set()
    sns.set_context('paper')
    sns.set_style('white')
    # matplotlib.rcParams['axes.linewidth'] = 0.5
    size = (8, 8)
    zlim = (-2, 1) if version == '1' else (-30, 80)
    offset = -2 if version == '1' else -30

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=size)
    surf = ax.plot_surface(X, Y, L, cmap=cm.coolwarm, linewidth=0)

        # Plot adjustments
    ax.contourf(X, Y, L, levels=contour_levels_loss, zdir='z', offset=offset, cmap=cm.coolwarm)
    ax.view_init(30, 255)
    ax.set(zlim=zlim, xlabel='True glucose concentration [mmol/L]', ylabel='Predicted glucose concentration [mmol/L]', zlabel='')
    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()

    filepath = os.path.join(save_dir, f'peg_loss_surface_v{version}')
    plt.savefig(filepath, bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()

    # Plot contour with PEG
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.contourf(X, Y, L, levels=contour_levels_loss, cmap=cm.coolwarm)
    format_error_grid(ax, grid_type='parkes', units='mmoll')
    ax.set(xlabel='True glucose concentration [mmol/L]', ylabel='Predicted glucose concentration [mmol/L]')

    filepath = os.path.join(save_dir, f'peg_loss_contour_v{version}')
    plt.savefig(filepath, bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()

    # Plot mu-derivative surface
    # L_y[np.abs(np.diff(L_y, axis=0, append=np.nan)) >= 0.1] = np.nan
    # L_y[np.abs(np.diff(L_y, axis=1, append=np.nan)) >= 0.1] = np.nan

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=size)
    surf = ax.plot_surface(X, Y, L_y, cmap=cm.coolwarm, linewidth=0)
    ax.contourf(X, Y, L_y, levels=contour_levels_derivative, zdir='z', offset=-4, cmap=cm.coolwarm)
    ax.view_init(30, 255)
    ax.set(xlabel='True glucose concentration [mmol/L]', ylabel='Predicted glucose concentration [mmol/L]', zlabel='')
    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()

    filepath = os.path.join(save_dir, f'peg_derivative_surface_v{version}')
    plt.savefig(filepath, bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()

    # Plot derivative contour with PEG
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.contourf(X, Y, L_y, levels=contour_levels_derivative, cmap=cm.coolwarm)
    format_error_grid(ax, grid_type='parkes', units='mmoll')
    ax.set(xlabel='True glucose concentration [mmol/L]', ylabel='Predicted glucose concentration [mmol/L]')

    filepath = os.path.join(save_dir, f'peg_derivative_contour_v{version}')
    plt.savefig(filepath, bbox_inches='tight', dpi=1200)
    plt.show()
    plt.close()


def plot_metric_over_epoch(logs, metric='loss', log=False, save_to=None, plot_show=True):
    df_logs = pd.DataFrame(logs).unstack().apply(pd.Series).stack().apply(pd.Series).stack()
    df_logs = df_logs.reset_index().rename(columns={'level_0': 'k',
                                                    'level_1': 'split',
                                                    'level_2': 'metric',
                                                    'level_3': 'epoch',
                                                    0: 'value'})
    df_logs = df_logs[df_logs.metric == metric]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.lineplot(data=df_logs, x='epoch', y='value', hue='split', ci='sd')
    plt.ylabel(metric)
    if log:
        plt.yscale('log')
        plt.ylabel('log ' + metric)
    plt.legend()

    if save_to:
        # plt.savefig(save_to + '.pdf', bbox_inches='tight')
        plt.savefig(save_to + '.png', bbox_inches='tight', dpi=600)
    if plot_show:
        plt.show()
    plt.close()


def plot_pointwise_metric(df,
                          metric,
                          save_to=None,
                          plot_show=True):
    plt.figure()
    sns.lineplot(data=df, x='point', y='value', hue='domain')
    plt.title('pointwise ' + metric)
    plt.xlabel('time [min]')
    plt.ylabel(metric)
    plt.legend()

    plt.savefig(save_to + '.pdf', bbox_inches='tight')
    plt.savefig(save_to + '.png', bbox_inches='tight', dpi=600)
    if plot_show:
        plt.show()
    plt.close()


def visualize_model_performance(logs, args):
    live_metrics_plots_path = os.path.join(args.base_dir, args.mtype, args.save_string, "plots", "live_metrics")
    if not os.path.exists(live_metrics_plots_path):
        os.makedirs(live_metrics_plots_path)

    plot_metric_over_epoch(logs, metric='loss', log=False, save_to=os.path.join(live_metrics_plots_path, 'loss'), plot_show=args.plot_show)
    plot_metric_over_epoch(logs, metric='loss', log=True, save_to=os.path.join(live_metrics_plots_path, 'log_loss'), plot_show=args.plot_show)

    plot_metric_over_epoch(logs, metric='RMSE', log=False, save_to=os.path.join(live_metrics_plots_path, 'RMSE'), plot_show=args.plot_show)
    plot_metric_over_epoch(logs, metric='RMSE', log=True, save_to=os.path.join(live_metrics_plots_path, 'log_RMSE'), plot_show=args.plot_show)


def visualize_loss_surface(x, y, loss):
    fig1, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    surf = ax.plot_surface(x, y, loss, cmap=cm.coolwarm, linewidth=1)
    ax.contourf(x, y, loss, zdir='z', offset=-5, cmap=cm.coolwarm)
    ax.view_init(30, 250)
    fig1.colorbar(surf, shrink=1, aspect=5)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.show()

    fig2, ax = plt.subplots(figsize=(10, 10))
    contour = ax.contourf(x, y, loss, cmap=cm.coolwarm, levels=9)
    fig2.colorbar(contour, shrink=1, aspect=5)
    plt.show()

    dzdy = np.gradient(loss.numpy(), 0.1)[1]
    fig3, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    surf = ax.plot_surface(x, y, dzdy, cmap=cm.coolwarm, linewidth=1)
    ax.contourf(x, y, dzdy, zdir='z', offset=0, cmap=cm.coolwarm)
    ax.view_init(30, 250)
    fig3.colorbar(surf, shrink=1, aspect=5)
    plt.show()

    fig4, ax = plt.subplots(figsize=(10, 10))
    contour = ax.contourf(x, y, dzdy, cmap=cm.coolwarm, levels=9)
    fig4.colorbar(contour, shrink=1, aspect=5)
    plt.show()
