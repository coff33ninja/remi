import math


def calculate(operation, num1, num2):
    try:
        num1, num2 = float(num1), float(num2)
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return "Error: Division by zero."
            return num1 / num2
        else:
            return "Unsupported operation."
    except ValueError:
        return "Error: Invalid numbers provided."
    except Exception as e:
        return f"Error: {str(e)}"


def advanced_calculate(equation):
    return "Advanced calculations not implemented yet."


def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    try:
        R = 6371  # Radius of Earth in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    except Exception as e:
        return f"Error calculating distance: {str(e)}"
