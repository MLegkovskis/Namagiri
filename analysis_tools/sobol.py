import openturns as ot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def perform_sobol_analysis(user_code, sobol_samples):
    """
    Executes the user-defined model code and performs Sobol analysis.
    Returns a dictionary with markdown-formatted results and raw values for plotting.
    """
    def run_user_code(user_code):
        # Create a persistent namespace for the user's code
        if not hasattr(run_user_code, "namespace"):
            run_user_code.namespace = {}
        exec(user_code, run_user_code.namespace)
        return run_user_code.namespace['model'], run_user_code.namespace['problem']

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
    # (Assuming the model code sets problem and that problem.getMarginal(i).getDescription() returns a list.)
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

    # For this example, add a placeholder for the radial plot description.
    radial_plot_description = "Radial plot description is not available in this analysis."

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
      - Right panel: A radial plot of S1 and ST indices.
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
    # Plot the S1 and ST indices with error bars.
    indices = df[["S1", "ST"]]
    err = df[["S1_conf", "ST_conf"]]
    # Use pandas plot.bar (the yerr expects a 2D array with shape matching the number of bars).
    indices.plot.bar(yerr=err.values.T, capsize=5, ax=ax1)
    ax1.set_title(f"Sobol Sensitivity Indices (N = {N})")
    ax1.set_ylabel("Sensitivity Index")
    ax1.set_xlabel("Input Variables")
    ax1.legend(["First-order", "Total-order"])

    # Right subplot: Radial plot.
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    _plot_sobol_radial(input_names, {"S1": S1, "ST": ST}, ax2)
    fig.tight_layout()
    return fig

def _plot_sobol_radial(names, indices, ax):
    """
    Helper function to generate a simple radial plot of S1 and ST indices.
    Plots the indices on a polar coordinate system.
    """
    N = len(names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Close the circle by repeating the first element.
    angles = np.concatenate((angles, [angles[0]]))
    S1 = np.concatenate((indices["S1"], [indices["S1"][0]]))
    ST = np.concatenate((indices["ST"], [indices["ST"][0]]))
    ax.plot(angles, S1, 'o-', label="First Order")
    ax.plot(angles, ST, 'o-', label="Total Order")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names)
    ax.set_title("Radial Plot of Sobol Indices")
    ax.legend(loc='upper right')
