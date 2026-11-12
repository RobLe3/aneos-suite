"""Display missing numerical inference without converting it to zero."""

def format_optional_number(value, precision=3):
    return 'unavailable' if value is None else f'{value:.{precision}f}'
