from .pnp_edm import PnPEDM, PnPEDMBatch

def get_sampler(config, model, kdiff, operator, noiser, device):
    if config.name == 'pnp_edm':
        return PnPEDM(config, model, kdiff, operator, noiser, device)
    elif config.name == 'pnp_edm_batch':
        return PnPEDMBatch(config, model, kdiff, operator, noiser, device)
    else:
        raise NameError(f"Model {config.name} is not defined.")

