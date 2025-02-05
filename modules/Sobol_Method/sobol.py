import openturns as ot
import numpy as np
import pandas as pd

def perform_sobol_analysis(user_code, sobol_samples):
    """
    Executes the user-defined model code and performs Sobol analysis.
    Returns a dictionary with markdown-formatted results.
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
    S2_matrix = sensitivity_analysis.getSecondOrderIndices()

    # Retrieve input parameter names
    input_names = [problem.getMarginal(i).getDescription()[0] for i in range(dimension)]

    # Prepare DataFrames for first and total order indices
    first_order_df = pd.DataFrame({
        'Variable': input_names,
        'First Order': S1
    })
    total_order_df = pd.DataFrame({
        'Variable': input_names,
        'Total Order': ST
    })

    # Prepare second order indices data
    second_order_data = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            value = float(S2_matrix[i, j])
            second_order_data.append({
                'Pair': f"{input_names[i]} - {input_names[j]}",
                'Second Order': value
            })
    second_order_df = pd.DataFrame(second_order_data)

    # Generate a description for the Sobol Indices Radial Plot
    radial_plot_description = (
        "A radial plot depicts the Sobol Indices with each variable represented as a segment on a circle. "
        "The length of the segment corresponds to the magnitude of the Sobol index, highlighting the relative "
        "importance of each input parameter."
    )

    # Generate a description of the uncertain input distributions
    inputs_description = ""
    for i in range(problem.getDimension()):
        marginal = problem.getMarginal(i)
        param_name = marginal.getDescription()[0]
        inputs_description += f"{param_name}: {marginal.__class__.__name__}, parameters {marginal.getParameter()}\n"

    # Return all results as markdown-formatted strings
    results = {
        "model_code_formatted": user_code,
        "inputs_description": inputs_description,
        "first_order_df": first_order_df.to_markdown(index=False),
        "total_order_df": total_order_df.to_markdown(index=False),
        "second_order_md_table": second_order_df.to_markdown(index=False) if not second_order_df.empty else "No second-order indices",
        "radial_plot_description": radial_plot_description
    }
    return results
