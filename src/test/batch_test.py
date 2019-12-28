import argparse


def test0(args):
    print("test0")
    print(f"a : {args.a}")
    print(f"b : {args.b}")
    if args.c_0 is not None:
        print(f"c_0 : {args.c_0}")


def test1(args):
    input_cmd = input()
    print("test0")
    print(f"a : {args.a}")
    print(f"b : {args.b}")
    if args.c_0 is not None:
        print(f"c_0 : {args.c_1}")
    print(input_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("a", type=str)
    parent_parser.add_argument("b", type=str)

    test0_parser = subparsers.add_parser("test0", parents=[parent_parser])
    test0_parser.add_argument("--c_0")
    test0_parser.set_defaults(handler=test0)

    test1_parser = subparsers.add_parser("test1", parents=[parent_parser])
    test1_parser.add_argument("--c_1")
    test1_parser.set_defaults(handler=test1)

    args = parser.parse_args()

    if hasattr(args, "handler"):
        args.handler(args)