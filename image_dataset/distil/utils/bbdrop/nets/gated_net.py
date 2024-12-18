from distil.utils.bbdrop.modules.gated_layers import *

class GatedNet(nn.Module):
    def __init__(self):
        super(GatedNet, self).__init__()
        self.gated_layers = []
        self.use_gate = False
        self.full_flops = 1.
        self.full_mem = 1.

    def build_gate(self, gate_fn, argdicts={}):
        self.use_gate = True
        if isinstance(argdicts, dict):
            argdicts =  [argdicts]*len(self.gated_layers)
        for i, layer in enumerate(self.gated_layers):
            layer.build_gate(gate_fn, **argdicts[i])

    def build_gate_dep(self, gate_fn, argdicts={}):
        self.use_gate = True
        if isinstance(argdicts, dict):
            argdicts =  [argdicts]*len(self.gated_layers)
        for i, layer in enumerate(self.gated_layers):
            layer.build_gate_dep(gate_fn, **argdicts[i])

    def reset_bb(self):
        for i, layer in enumerate(self.gated_layers):
            layer.gate.reset_params()

    def reset_dep(self):
        for i, layer in enumerate(self.gated_layers):
            layer.dgate.reset()
            
    def get_params(self):
        # for i, layer in enumerate(self.gated_layers):
        print(self.gated_layers[-1].gate.get_params())
            
    def stop_training(self):
        for i, layer in enumerate(self.gated_layers):
            layer.gate.stop_training()

    def get_reg(self):
        reg = 0.
        for layer in self.gated_layers:
            reg += layer.get_reg()
        return reg

    def get_reg_dep(self):
        reg = 0.
        for layer in self.gated_layers:
            reg += layer.get_reg_dep()
        return reg

    def get_pruned_size(self):
        return [layer.get_num_active() for layer in self.gated_layers]

    def get_pruned_size_dep(self):
        return [int(layer.dgate.num_active) for layer in self.gated_layers]

    def count_flops(self, num_units):
        raise NotImplementedError

    def count_flops_dep(self, num_units, num_units_dep):
        raise NotImplementedError

    def count_memory(self, num_units):
        raise NotImplementedError

    def count_memory_dep(self, num_units, num_units_dep):
        raise NotImplementedError

    def get_speedup(self):
        pruned = self.get_pruned_size()
        return float(self.full_flops) / float(self.count_flops(pruned))

    def get_speedup_dep(self):
        pruned = self.get_pruned_size()
        pruned_dep = self.get_pruned_size_dep()
        return float(self.full_flops) / \
                float(self.count_flops_dep(pruned, pruned_dep))

    def get_memory_saving(self):
        pruned = self.get_pruned_size()
        return float(self.count_memory(pruned)) / float(self.full_mem)

    def get_memory_saving_dep(self):
        pruned = self.get_pruned_size()
        pruned_dep = self.get_pruned_size_dep()
        return float(self.count_memory_dep(pruned, pruned_dep)) / \
                float(self.full_mem)
