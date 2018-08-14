import re
import sys


def rename_imports(prefix, file):
    """
    Fix the eos imports in the file to add the given prefix.
    For the _eos import, the alias 'as _eos' is added to the end of the line so
    that the calls to the _eos library still work.
    """
    p = re.compile('import (eos\S*)')
    p2 = re.compile('import (_eos\S*)')
    read_data = []

    with open(file, 'r') as f:
        read_data = f.readlines()

    with open(file, 'w') as f:
        if (file == 'eos/__init__.py') and read_data != [] and read_data[1][0:3] != '__a':
            f.write("__all__ = ['eos_module', 'eos_type_module', 'network']\n")
            read_data = read_data[1:]
        for l in read_data:
            m = p.match(l)
            m2 = p2.match(l)
            if m:
                f.write("import {}.{}\n".format(prefix, m.group(1)))
            elif m2:
                f.write("import {}.{} as _eos\n".format(prefix, m2.group(1)))
            else:
                f.write(l)


if __name__ == "__main__":
    for arg in sys.argv[2:]:
        rename_imports(sys.argv[1], arg)
