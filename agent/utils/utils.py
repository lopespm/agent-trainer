class LinearInterpolator(object):
    def interpolate(self, x, x0, x1, y0, y1):
        x = float(x)
        x0 = float(x0)
        x1 = float(x1)
        y0 = float(y0)
        y1 = float(y1)
        return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

    def interpolate_with_clip(self, x, x0, x1, y0, y1):
        interpolation = self.interpolate(x=x, x0=x0, x1=x1, y0=y0, y1=y1)
        return min(y0, max(y1, interpolation))


class NumberHistory(object):
    def __init__(self):
        self.current_number = 0.0
        self.previous_number = None

    def add(self, number):
        self.previous_number = self.current_number
        self.current_number = number

    def current(self):
        return self.current_number

    def previous(self):
        return self.previous_number