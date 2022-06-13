from run_nerf import create_instantngp, create_nerf, create_nerfwithhash


def load(args):
    # Create nerf model
    create_method = create_nerf
    if args.hashenc:
        create_method = create_nerfwithhash
    elif args.instant_ngp:
        create_method = create_instantngp
        
    return create_method(args)