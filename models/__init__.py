from models import nerf
from models import instant_ngp

def load(args):
    # Create nerf model
    create_method = nerf.create
    if args.instant_ngp:
        create_method = instant_ngp.create

    return create_method(args)