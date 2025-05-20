def calculator(expression: str) -> str:
    """
    Calculate the result of the arithmetic expression.

    Args:
        expression: Expression to calculate.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for, in the format "city, country"
    """
    return 22.0

