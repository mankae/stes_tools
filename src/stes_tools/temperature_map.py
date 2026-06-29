import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def temperature_map_simple(
    n_layers=32, # layers in the storage = number of temperature sensors in the storage
    T_min= 12, # minimum temperature in the storage, based on the Dronninglund data
    T_max=90, # maximum temperature in the storage, based on the Dronninglund data
    n_points=2000, # points in the synthetic temperature curves
    stretch=1, # stretch factor for how different the curves are
    sharpness=20, # sharpness of the temperature curves, higher values lead to steeper curves
    plot=False # whether to plot the generated temperature curves, default is False
    ):
    
    x = np.linspace(0, 1, n_points)
    
    # shift controls where the curve bends
    shifts = np.linspace(-0.5 * stretch, 0.5 * stretch, n_layers)

    curves = []
    for s in shifts:
        y = 1 / (1 + np.exp(-sharpness * (x - 0.5 + s)))
        # normalize to (0,0) → (1,1)
        y = (y - y.min()) / (y.max() - y.min())
        curves.append(y)
    
    curves = np.array(curves) * (T_max - T_min) + T_min
    curves = np.flip(curves, axis=0) # flip to have the coldest curve at the bottom and the hottest at the top
    
    # --- optional plot ---
    if plot:
        
        plt.rcParams.update({'font.size': 19})

        plt.figure(figsize=(10, 5))
        # for i in range(n_layers):
            # plt.plot(x, curves[i], color=plt.cm.rainbow(i / n_layers), alpha=0.9)

        # Custom 3-color gradient
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_brown",
            ["#833C0C", "#ED7D31", "#FCE4D6"]
        )

        for i in range(n_layers):
            plt.plot(x * 100, curves[i], color=cmap(i / (n_layers - 1)), linewidth=3, alpha=1.0)

        plt.xlabel(r"$Q_t$ [%]", fontsize=21)
        plt.ylabel(r"$T_n$ [°C]", fontsize=21)
        # plt.title("Synthetic Layered Temperature Curves")
        plt.show()
        plt.rcParams.update({'font.size': 10})

    return np.array(curves)

def temperature_map(
    n_layers=32,
    T_min=12,
    T_mid=45,       # set to None to skip the cascaded stage and use a single T_min→T_max curve
    T_max=90,
    n_points=2000,
    stretch=1,
    sharpness=10,
    overlap=0.3,
    plot=False
    ):

    if T_mid is None:
        # --- Single-stage fallback: behaves like v2 ---
        T_curves = temperature_map_simple(
            n_layers, T_min, T_max, n_points, stretch, sharpness, False
        )

    else:
        # --- Two-stage cascaded curves (original logic) ---
        shift = int(n_points * (1 - overlap))

        curves_1 = temperature_map_simple(n_layers, T_min, T_mid, n_points, stretch, sharpness, False)
        curves_2 = temperature_map_simple(n_layers, 0, T_max - T_mid, n_points, stretch, sharpness, False)
        curves_3 = temperature_map_simple(n_layers, T_min, T_max, n_points + shift, stretch, sharpness - 4, False)

        shift_zeros  = np.zeros((n_layers, shift))
        shift_T_mid  = np.column_stack([curves_1[:, -1]] * shift)

        curves_1 = np.hstack((curves_1, shift_T_mid))
        curves_2 = np.hstack((shift_zeros, curves_2))
        curves_2 += T_mid

        transition_1 = np.linspace(-np.pi/2, np.pi/2, n_points - shift)
        transition_1 = (np.sin(transition_1) + 1) / 2
        mask_1 = np.hstack((np.ones(shift), 1 - transition_1, np.zeros(shift)))
        mask_2 = np.hstack((np.zeros(shift), transition_1, np.ones(shift)))

        shift_transition_2 = int(n_layers / 2)
        transition_2 = np.linspace(-np.pi/2, np.pi/2, shift_transition_2)
        transition_2 = (np.sin(transition_2) + 1) / 2
        transition_2 = np.hstack((np.zeros(n_layers - shift_transition_2), transition_2))
        transition_2 = np.flip(transition_2)
        mask_3 = np.column_stack([transition_2.T] * len(mask_1))

        T_curves = (curves_1 * mask_1 + curves_2 * mask_2) * (1 - mask_3) + curves_3 * mask_3

    # --- Optional plot ---
    if plot:
        plt.rcParams.update({'font.size': 19})

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_brown",
            ["#833C0C", "#ED7D31", "#FCE4D6"]
        )

        plt.figure(figsize=(10, 5))
        for i in range(n_layers):
            x = np.linspace(0, 1, T_curves.shape[1]) * 100
            plt.plot(x, T_curves[i], color=cmap(i / (n_layers - 1)), linewidth=3, alpha=0.9)

        plt.xlabel(r"$Q_t$ [%]", fontsize=21)
        plt.ylabel(r"$T_n$ [°C]", fontsize=21)
        plt.show()
        plt.rcParams.update({'font.size': 10})

    return T_curves