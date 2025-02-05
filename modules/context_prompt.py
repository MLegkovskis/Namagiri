def build_context_prompt(user_message, module_context):
    """
    Aggregates context from various modules to build a prompt for the LLM.
    The module_context should be a dictionary where each key is a module name
    (e.g. 'Sobol') and the value is the corresponding context dictionary.
    """
    if module_context:
        prompt_parts = []
        for module, context in module_context.items():
            if context:
                if module.lower() == "sobol":
                    part = f"""--- Sobol Analysis Context ---
First-order Sobol indices:
{context.get('first_order_df', '')}

Total-order Sobol indices:
{context.get('total_order_df', '')}

Second-order Sobol indices:
{context.get('second_order_md_table', '')}
"""
                    prompt_parts.append(part)
                else:
                    # Placeholder for other modules, such as Shapley, etc.
                    part = f"--- {module} Context ---\n{context}\n"
                    prompt_parts.append(part)
        if prompt_parts:
            context_prompt = "\n".join(prompt_parts)
            final_prompt = (
                f"User question: {user_message}\n\n"
                f"Additional context from modules:\n\n{context_prompt}\n\n"
                "Please provide a detailed explanation regarding the above context in relation to the user query."
            )
            return final_prompt
    return user_message
