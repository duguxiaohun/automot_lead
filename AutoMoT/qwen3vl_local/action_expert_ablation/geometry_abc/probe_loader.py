"""Keep the configured sensor worker processes alive through the GPU probe."""
import copy


def close(loader):
    # Same PyTorch iterator shutdown protocol used by the shared training loop.
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator=None


def prepare(chosen,args):
    from .loader import make_loader
    local=copy.copy(args)
    local.persistent_workers=local.num_workers>0
    count=max(1,local.num_workers)
    loader,_=make_loader([chosen]*count,local,rank=0,world_size=1,shuffle=False,epoch_seed=args.seed)
    first=None;decoded=0
    try:
        # One CPU decode per worker forces every spawned worker to finish
        # initialization. Repeated decodes are not extra training exposures.
        for prepared in loader:
            if prepared.get('_error'):raise ValueError(prepared['_error'])
            if prepared['sample']['abc_record']['id']!=chosen['abc_record']['id']:
                raise ValueError('probe_loader_member_changed')
            if first is None:first=prepared
            decoded+=1
        if decoded!=count:raise ValueError('probe_loader_incomplete')
        return loader,first,dict(configured_workers=local.num_workers,decoded_copies=decoded,
            formal_training_exposures=0,lifetime='through checkpoint reload')
    except BaseException:
        close(loader)
        raise
