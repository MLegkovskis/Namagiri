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

Given a description of the Sobol Indices Radial Plot data:

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
