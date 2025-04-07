import math


def calculate(operation, num1, num2):
    try:
        num1, num2 = float(num1), float(num2)
        if operation == "add" or operation == "+":
            return f"{num1} + {num2} = {num1 + num2}"
        elif operation == "subtract" or operation == "-":
            return f"{num1} - {num2} = {num1 - num2}"
        elif operation == "multiply" or operation == "*" or operation == "times":
            return f"{num1} * {num2} = {num1 * num2}"
        elif operation == "divide" or operation == "/":
            if num2 == 0:
                return "Error: Division by zero!"
            return f"{num1} / {num2} = {num1 / num2}"
        else:
            return f"Unsupported operation: {operation}"
    except ValueError:
        return "Error: Please provide valid numbers."
    except Exception as e:
        return f"Calculation error: {str(e)}"


def advanced_calculate(equation):
    return "Advanced calculations not implemented yet."


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula."""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance
    except Exception as e:
        return f"Distance calculation error: {str(e)}"
