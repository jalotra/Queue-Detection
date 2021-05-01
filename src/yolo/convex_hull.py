import math

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def cmp(a : Point, b : Point) -> bool:
        return a.x < b.x or (a.x == b.x and (a.y < b.y))

def clockwise(a : Point, b : Point, c : Point) -> bool:
    return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) < 0
    
def counter_clockwise(a : Point, b : Point, c : Point) -> bool:
    return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) > 0

class ConvexHull():
    def __init__(self, points):
        self.points = []
        for x in points:
            assert(len(x) == 2)
            self.points.append(Point(x[0], x[1]))
        # print(self.points)
    def find_convex_hull(self):
        if(len(self.points) == 1):
            return 
        # Have to think how to do this
        self.points = sorted(self.points, key = lambda p : (p.x, p.y))
        p1 = self.points[0]; p2 = self.points[-1]

        up, down = [], [] 
        up.append(p1);down.append(p1)

        for i in range(1, len(self.points)):
            if(i == len(self.points) - 1 or clockwise(p1, self.points[i], p2)):
                while(len(up) >= 2 and not clockwise(up[len(up) - 2], up[len(up) - 1], self.points[i])):
                    up.pop()
                up.append(self.points[i])

            if(i == len(self.points) - 1 or counter_clockwise(p1, self.points[i], p2)):
                while(len(down) >= 2 and not counter_clockwise(down[len(down) - 2], down[len(down) - 1], self.points[i])):
                    down.pop()
                down.append(self.points[i])

        self.points = []
        for a in up:
            self.points.append(a)
        for b in down:
            self.points.append(b)
        
        return self.points