from .edm_kernel.edm_kernel import create_edm_from_kernel_unet

def get_model(name: str, **kwargs):
    if name == 'edm_from_kernel_unet':
        return create_edm_from_kernel_unet(**kwargs)
    else:
        raise NameError(f"Model {name} is not defined.")
