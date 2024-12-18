def count_memory_dense(in_features, out_features, bias=True, batch_norm=False):
    mem = in_features * out_features
    if bias:
        mem += out_features
    if batch_norm:
        mem += 2 * in_features
    return mem

def count_memory_conv(height, width, in_channels, out_channels, kernel_size,
        stride=1, padding=0, bias=True, batch_norm=False):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*2
    n = kernel_size[0] * kernel_size[1] * in_channels
    out_height = (height - kernel_size[0] + 2*padding) / stride + 1
    out_width = (width - kernel_size[1] + 2*padding) / stride + 1
    mem_fmap = n * out_height * out_width
    mem_kernel = n * out_channels
    mem = mem_fmap + mem_kernel
    if batch_norm:
        mem += 2 * out_channels
    return mem

def count_memory_dbb(num_gates):
    return 4 * num_gates
