class PID(object):
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.previous_error = 0
        self.cumulative_error = 0

    def get_action(self, error):
        action = self.compute_action(input_p=error, input_i=self.cumulative_error, input_d=error-self.previous_error)
        self.previous_error = error
        self.cumulative_error += error
        return action

    def compute_action(self, input_p, input_i, input_d):
        return self.k_p*input_p + self.k_i*input_i + self.k_d*input_d
