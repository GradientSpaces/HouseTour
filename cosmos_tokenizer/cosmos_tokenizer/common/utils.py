import os
import re

def get_abs_path(rel_path):
    pass
    # TODO: Implement the library_root without using registry
    #return os.path.join(registry.get_path("library_root"), rel_path)

def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url