import termcolor


def output_with_color(string, c="r"):
    width = len(string) + 10 if len(string) >= 30 else 30
    output = "-" * width + "\n"
    output += "|" + string.center(width - 2, ".") + "|" + "\n"
    output += "-" * width
    if c == "r":
        print(termcolor.colored(output, "red"))
    elif c == "y":
        print(termcolor.colored(output, "yellow"))
    elif c == "g":
        print(termcolor.colored(output, "green"))
