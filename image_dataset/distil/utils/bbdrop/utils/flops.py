def count_flops_dense(in_features, out_features,
        bias=True, activation=True):
    flops = (2*in_features-1)*out_features
    if bias:
        flops += out_features
    if activation:
        flops += out_features
    return flops

def count_flops_conv(height, width, in_channels, out_channels, kernel_size,
        stride=1, padding=0, bias=True, activation=True):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*2
    n = kernel_size[0] * kernel_size[1] * in_channels
    flops_per_instance = 2*n - 1
    out_height = (height - kernel_size[0] + 2*padding) / stride + 1
    out_width = (width - kernel_size[1] + 2*padding) / stride + 1
    num_instances_per_channel = out_height * out_width
    flops_per_channel = num_instances_per_channel * flops_per_instance
    total_flops = out_channels * flops_per_channel
    if bias:
        total_flops += out_channels * num_instances_per_channel
    if activation:
        total_flops += out_channels * num_instances_per_channel
    return total_flops

def count_flops_dense_dbb(num_gates):
    # mask construction + multiplication
    total_flops = 5 * num_gates
    return total_flops

def count_flops_conv_dbb(height, width, num_gates):
    # global avg pool
    total_flops = num_gates * height * width
    # mask construction
    total_flops += 4 * num_gates
    # mask multiplication
    total_flops += num_gates * height * width
    return total_flops

def count_flops_max_pool(height, width, channels, kernel_size,
        stride=None, padding=0):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*2
    stride = kernel_size if stride is None else stride
    if isinstance(stride, int):
        stride = [stride]*2
    flops_per_instance = kernel_size[0] * kernel_size[1]
    out_height = (height - kernel_size[0] + 2*padding) / stride[0] + 1
    out_width = (width - kernel_size[1] + 2*padding) / stride[1] + 1
    num_instances_per_channel = out_height * out_width
    flops_per_channel = num_instances_per_channel * flops_per_instance
    total_flops = channels * flops_per_channel
    return total_flops

def count_flops_global_avg_pool(height, width, channels):
    return channels * height * width
