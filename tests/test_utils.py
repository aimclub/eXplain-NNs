def _form_message_header(message_header=None):
    return 'Value mismatch' if message_header is None else message_header

def compare_values(expected, got, message_header=None):
    assert expected == got, f"{_form_message_header(message_header)}: expected {expected}, got {got}"