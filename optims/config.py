def normalize_cflat_args(args):
    mode = args.get("cflat_mode")
    if mode is None:
        mode = "cflat" if args.get("cflat", False) else "off"
    if mode not in {"off", "cflat", "cflat-plus"}:
        raise ValueError('"cflat_mode" should be one of ["off", "cflat", "cflat-plus"].')

    args["cflat_mode"] = mode
    args["cflat"] = mode != "off"
    for key, infty_key, default in (
        ("cflat_A", "A", 5.0),
        ("cflat_k", "k", 0.01),
        ("cflat_t0", "t0", 80),
        ("cflat_cof", "cof", 0.005),
    ):
        if args.get(key) is None:
            args[key] = args.get(infty_key, default)
