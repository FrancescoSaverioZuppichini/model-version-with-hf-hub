import hashlib
import json
from typing import Any, Dict, List


# gently copied from https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def uid_from_dictionary(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def markdown_table(header: List[str], rows: List[List[Any]]) -> str:
    num_cols = len(header)
    table = ""
    # header
    table += "|" + "|".join(header) + "|"
    table += "\n"
    table += "|" + "".join([" --- |"] * num_cols)
    # values
    for row in rows:
        table += "\n"
        table += "|" + "|".join(map(str, row)) + "|"

    return table
