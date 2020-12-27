class PID(object):
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def get_action(self, input_p, input_i, input_d):
        return self.k_p*input_p + self.k_i*input_i + self.k_d*input_d
