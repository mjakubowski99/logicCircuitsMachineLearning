import numpy as np

def random(min, max):
    rng = np.random.default_rng()
    return rng.integers(min,max)

def repeat_random_until_got_new_value(min, max, drawn):
    value = random(min, max)

    while value in drawn:
        value = random(min,max)

    return value
