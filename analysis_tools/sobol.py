# modules/analysis_tools/sobol.py
import openturns as ot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Analysis Functions ---

def perform_sobol_analysis(user_code, sobol_samples):
    """
    Executes the user-defined model code and performs Sobol analysis.
    Returns a dictionary with markdown-formatted results and raw values for plotting.
    """
    def run_user_code(user_code):
        # Use a fresh local namespace for each execution.
        local_namespace = {}
        exec(user_code, local_namespace)
        # Ensure that both 'model' and 'problem' are defined.
        if 'model' not in local_namespace:
            raise KeyError("The executed code did not define 'model'. Please ensure your code defines a variable named 'model'.")
        if 'problem' not in local_namespace:
            raise KeyError("The executed code did not define 'problem'. Please ensure your code defines a variable named 'problem'.")
        return local_namespace['model'], local_namespace['problem']

    model, problem = run_user_code(user_code)

    # Create an independent copy of the distribution for Sobol analysis
    marginals = [problem.getMarginal(i) for i in range(problem.getDimension())]
    independent_dist = ot.JointDistribution(marginals)
    dimension = independent_dist.getDimension()
    compute_second_order = True

    sie = ot.SobolIndicesExperiment(independent_dist, sobol_samples, compute_second_order)
    input_design = sie.generate()
    output_design = model(input_design)
    sensitivity_analysis = ot.SaltelliSensitivityAlgorithm(
        input_design, output_design, sobol_samples
    )

    # Extract first- and total-order indices
    S1 = [float(x) for x in sensitivity_analysis.getFirstOrderIndices()]
    ST = [float(x) for x in sensitivity_analysis.getTotalOrderIndices()]
    S2 = sensitivity_analysis.getSecondOrderIndices()  # 2D array

    # Compute confidence intervals (assumed approximately symmetric)
    S1_interval = sensitivity_analysis.getFirstOrderIndicesInterval()
    lower_bound = S1_interval.getLowerBound()
    upper_bound = S1_interval.getUpperBound()
    S1_conf = [(upper_bound[i] - lower_bound[i]) / 2.0 for i in range(dimension)]
    ST_interval = sensitivity_analysis.getTotalOrderIndicesInterval()
    lower_bound = ST_interval.getLowerBound()
    upper_bound = ST_interval.getUpperBound()
    ST_conf = [(upper_bound[i] - lower_bound[i]) / 2.0 for i in range(dimension)]

    # Retrieve input parameter names from the problem definition.
    # (Assuming that problem.getMarginal(i).getDescription() returns a list.)
    input_names = [problem.getMarginal(i).getDescription()[0] for i in range(dimension)]

    # Prepare DataFrames for reporting (converted to markdown later)
    first_order_df = pd.DataFrame({
        'Variable': input_names,
        'First Order': S1
    })
    total_order_df = pd.DataFrame({
        'Variable': input_names,
        'Total Order': ST
    })
    # Process second-order indices for reporting (only significant interactions)
    second_order_data = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            value = float(S2[i, j])
            if not np.isnan(value) and abs(value) > 0.01:
                second_order_data.append({
                    'Pair': f"{input_names[i]} - {input_names[j]}",
                    'Second Order': value
                })
    second_order_df = pd.DataFrame(second_order_data)

    # Generate a description of the uncertain input distributions
    inputs_description = ""
    for i in range(problem.getDimension()):
        marginal = problem.getMarginal(i)
        param_name = marginal.getDescription()[0]
        inputs_description += f"{param_name}: {marginal.__class__.__name__}, parameters {marginal.getParameter()}\n"

    # Set the radial plot description using our new function.
    radial_plot_description = get_radial_plot_description()

    results = {
        "model_code_formatted": user_code,
        "inputs_description": inputs_description,
        "first_order_df": first_order_df.to_markdown(index=False),
        "total_order_df": total_order_df.to_markdown(index=False),
        "second_order_md_table": second_order_df.to_markdown(index=False) if not second_order_df.empty else "No significant second-order interactions detected.",
        "radial_plot_description": radial_plot_description,
        # Raw data for plotting
        "S1": S1,
        "ST": ST,
        "S2": S2,
        "S1_conf": S1_conf,
        "ST_conf": ST_conf,
        "input_names": input_names
    }
    return results

def build_sobol_prompt(sobol_results):
    """
    Builds the prompt for the LLM using the results from the Sobol analysis.
    """
    RETURN_INSTRUCTION = "Please interpret the following Sobol analysis results."
    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{sobol_results["model_code_formatted"]}
```

and the following uncertain input distributions:

{sobol_results["inputs_description"]}

Given the following first-order Sobol' indices and their confidence intervals:

{sobol_results["first_order_df"]}

And the following total-order Sobol' indices and their confidence intervals:

{sobol_results["total_order_df"]}

The following second-order Sobol' indices were identified:

{sobol_results["second_order_md_table"]}

An interpretation of the Sobol Indices Radial Plot is provided:

{sobol_results["radial_plot_description"]}

Please:
  - Categorise the Sobol' indices as weak (index < 0.05), moderate (0.05 ≤ index ≤ 0.2) or strong (index > 0.2).
  - Display all index values as separate tables (if tables are large, show only the top 10 ranked inputs).
  - Briefly explain the Sobol method and the difference between first-order and total-order indices in terms of their mathematics and significance.
  - Explain the significance of high-impact Sobol' indices and the importance of the corresponding input variables from both mathematical and physical perspectives.
  - Discuss the confidence intervals associated with the Sobol' indices.
  - Provide an interpretation of the Sobol Indices Radial Plot based on the description and numerical data.
  - Reference the Sobol indices tables in your discussion.
    """
    return prompt

def plot_sobol_indices(sobol_results, N="?", second_order_index_threshold=0.05):
    """
    Generates a combined figure with:
      - Left panel: Bar plot of first-order (S1) and total-order (ST) Sobol indices with error bars,
        using S1_conf and ST_conf.
      - Right panel: A radial plot of S1 and ST indices using the updated radial plot code.
    The x-ticks are labeled using the input variable names.
    
    Parameters:
      - sobol_results: dict returned by perform_sobol_analysis.
      - N: The sample size used (to be displayed in the title).
      - second_order_index_threshold: A threshold used in the radial plot description (if needed).
    
    Returns:
      A Matplotlib figure.
    """
    S1 = np.array(sobol_results.get("S1"))
    ST = np.array(sobol_results.get("ST"))
    S1_conf = np.array(sobol_results.get("S1_conf"))
    ST_conf = np.array(sobol_results.get("ST_conf"))
    input_names = sobol_results.get("input_names")
    
    if S1 is None or ST is None or input_names is None:
        return None

    # Create a DataFrame for plotting the bar chart.
    data = {
        "S1": S1,
        "ST": ST,
        "S1_conf": S1_conf,
        "ST_conf": ST_conf
    }
    df = pd.DataFrame(data, index=input_names)

    # Create a figure with two subplots.
    fig = plt.figure(figsize=(16, 6))
    # Left subplot: Bar plot with error bars.
    ax1 = fig.add_subplot(1, 2, 1)
    indices = df[["S1", "ST"]]
    err = df[["S1_conf", "ST_conf"]]
    indices.plot.bar(yerr=err.values.T, capsize=5, ax=ax1)
    ax1.set_title(f"Sobol Sensitivity Indices (N = {N})")
    ax1.set_ylabel("Sensitivity Index")
    ax1.set_xlabel("Input Variables")
    # Explicitly set the x-tick labels to the variable names with rotation for readability.
    ax1.set_xticklabels(df.index, rotation=45, ha="right")
    ax1.legend(["First-order", "Total-order"])

    # Right subplot: Radial plot using the new, expert-verified code.
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    plot_sobol_radial(input_names, {"S1": S1, "ST": ST, "S2": np.array(sobol_results.get("S2"))},
                      ax2, sensitivity_threshold=0.01, max_marker_radius=70.0, tolerance=0.1)
    fig.tight_layout()
    return fig

# --- New Helper Functions for Radial Plotting ---

def clip_01(x):
    """Clips x to the interval [0, 1]."""
    return max(0, min(1, x))

def plot_sobol_radial(variable_names, sobol_indices, ax, sensitivity_threshold=0.01,
                      max_marker_radius=70.0, tolerance=0.1):
    """
    Plot Sobol indices on a radial plot.

    The Sobol Indices Radial Plot is a polar plot where each input variable
    is placed at equal angular intervals around a circle.

    Parameters
    ----------
    variable_names : list of str
        The list of input variable names.
    sobol_indices : dict
        A dictionary with keys:
            - "S1": 1D np.array of first-order Sobol' indices.
            - "ST": 1D np.array of total-order Sobol' indices.
            - "S2": 2D np.array of second-order Sobol' indices.
    ax : matplotlib axis
        The polar axis on which to plot.
    sensitivity_threshold : float
        Threshold below which indices are considered insignificant.
    max_marker_radius : float
        The maximum marker radius in points.
    tolerance : float
        Tolerance for checking that indices are in [0,1].
    """
    # Check indices dimensions
    dimension = len(variable_names)
    if len(sobol_indices["S1"]) != dimension:
        raise ValueError(f"The number of variable names is {dimension} but the number of first-order Sobol' indices is {len(sobol_indices['S1'])}.")
    if len(sobol_indices["ST"]) != dimension:
        raise ValueError(f"The number of variable names is {dimension} but the number of total order Sobol' indices is {len(sobol_indices['ST'])}.")
    if sobol_indices["S2"].shape != (dimension, dimension):
        raise ValueError(f"The number of variable names is {dimension} but the shape of second order Sobol' indices is {sobol_indices['S2'].shape}.")

    # Check indices values and clip if necessary
    S1 = list(sobol_indices["S1"])  # make mutable copy
    ST = list(sobol_indices["ST"])
    S2 = sobol_indices["S2"].copy()
    for i in range(dimension):
        if S1[i] < -tolerance or S1[i] > 1.0 + tolerance:
            print(f"Warning: The first-order Sobol' index of variable #{i} is {S1[i]} which is not in [0,1], up to the tolerance {tolerance}.")
        S1[i] = clip_01(S1[i])
        if ST[i] < -tolerance or ST[i] > 1.0 + tolerance:
            print(f"Warning: The total order Sobol' index of variable #{i} is {ST[i]} which is not in [0,1], up to the tolerance {tolerance}.")
        ST[i] = clip_01(ST[i])
        for j in range(i + 1, dimension):
            if S2[i, j] < -tolerance or S2[i, j] > 1.0 + tolerance:
                print(f"Warning: The second order Sobol' index of variables ({i}, {j}) is {S2[i, j]} which is not in [0,1], up to the tolerance {tolerance}.")
            S2[i, j] = clip_01(S2[i, j])
    # Convert S1 and ST to NumPy arrays so that elementwise comparison is supported.
    S1 = np.array(S1)
    ST = np.array(ST)

    # Filter out insignificant indices based on ST
    significant = ST > sensitivity_threshold
    significant_names = np.array(variable_names)[significant]
    significant_dimension = len(significant_names)
    significant_angles = np.linspace(0, 2 * np.pi, significant_dimension, endpoint=False)
    ST_sig = ST[significant]
    S1_sig = S1[significant]

    # Prepare S2 matrix for significant indices
    S2_matrix = np.zeros((len(significant_names), len(significant_names)))
    for i in range(len(significant_names)):
        for j in range(i + 1, len(significant_names)):
            idx_i = np.where(np.array(variable_names) == significant_names[i])[0][0]
            idx_j = np.where(np.array(variable_names) == significant_names[j])[0][0]
            S2_value = S2[idx_i, idx_j]
            if np.isnan(S2_value) or S2_value < sensitivity_threshold:
                S2_value = 0.0
            S2_matrix[i, j] = S2_value

    # Plotting on polar axis
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(significant_angles)
    ax.set_xticklabels([str(name) for name in significant_names])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.5)

    # Plot total-order (ST) circles (outer circles)
    for loc, st_val in zip(significant_angles, ST_sig):
        s = st_val * max_marker_radius**2
        ax.scatter(loc, 1, s=s, c="white", edgecolors="black", zorder=2)
    # Plot first-order (S1) circles (inner circles)
    for loc, s1_val in zip(significant_angles, S1_sig):
        if s1_val > sensitivity_threshold:
            s = s1_val * max_marker_radius**2
            ax.scatter(loc, 1, s=s, c="black", edgecolors="black", zorder=3)

    # Plot second-order interactions (S2) as lines
    from matplotlib.lines import Line2D
    for i in range(len(significant_names)):
        for j in range(i + 1, len(significant_names)):
            if S2_matrix[i, j] > 0:
                xi = np.cos(significant_angles[i])
                xj = np.cos(significant_angles[j])
                yi = np.sin(significant_angles[i])
                yj = np.sin(significant_angles[j])
                distance_ij = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                lw = S2_matrix[i, j] * max_marker_radius / distance_ij
                ax.plot([significant_angles[i], significant_angles[j]], [1, 1],
                        c="darkgray", lw=lw, zorder=1)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="ST",
               markerfacecolor="white", markeredgecolor="black", markersize=15),
        Line2D([0], [0], marker="o", color="w", label="S1",
               markerfacecolor="black", markeredgecolor="black", markersize=15),
        Line2D([0], [0], color="darkgray", lw=3, label="S2"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plot_title = "Sobol Indices"
    ax.set_title(plot_title)
    return

def get_radial_plot_description():
    description = f"""
**Radial Plot Description**
The Sobol Indices Radial Plot is a polar plot where each input variable is placed at equal angular intervals around a circle. The elements of the plot are:

- **Variables**: Each input variable is positioned at a specific angle on the circle, equally spaced from others.

- **Circles**:
    - The **outer circle** (white fill) represents the **total-order Sobol' index (ST)** for each variable.
    - The **inner circle** (black fill) represents the **first-order Sobol' index (S1)**.
    - The **area of the circles** is proportional to the magnitude of the respective Sobol' indices.

- **Lines**:
    - Lines connecting variables represent **second-order Sobol' indices (S2)**.
    - The **area of the lines** corresponds to the magnitude of the interaction between the two variables; thicker lines indicate stronger interactions.

This plot visually conveys both the individual effects of variables and their interactions, aiding in understanding the model's sensitivity to input uncertainties.
"""
    return description

def problem_to_python_code(problem):
    distributions_formatted = ',\n        '.join(
        [f"{dist}" for dist in problem['distributions']]
    )

    code = f'''problem = {{
    'num_vars': {problem['num_vars']},
    'names': {problem['names']},
    'distributions': [
        {distributions_formatted}
    ]
}}
'''
    return code

def describe_radial_plot(Si, variable_names, sensitivity_threshold=0.01):
    radial_data = ""
    radial_data += f"\nThreshold for significant Sobol' indices is {sensitivity_threshold}.\n"
    for i, name in enumerate(variable_names):
        s1_value = Si["S1"][i]
        st_value = Si["ST"][i]
        radial_data += f"- Variable **{name}**: S1 = {s1_value:.4f}, ST = {st_value:.4f}\n"

    # Count number of significant interactions and list them
    number_of_significant_interactions = 0
    dimension = len(variable_names)
    for i in range(dimension):
        for j in range(i + 1, dimension):
            s2_value = Si["S2"][i, j]
            if s2_value > sensitivity_threshold:
                number_of_significant_interactions += 1
                radial_data += f"- Interaction between **{variable_names[i]}** and **{variable_names[j]}**: S2 = {s2_value:.4f}\n"

    if number_of_significant_interactions == 0:
        radial_data += "\nNo significant second-order interactions detected."

    radial_plot_description = f"""
{get_radial_plot_description()}

Numerical data for the plot:
{radial_data}
"""
    return radial_plot_description
