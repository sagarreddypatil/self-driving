def apply_deadzone(input):
    if abs(input) <= 0.075:
        return 0.0
    else:
        return input