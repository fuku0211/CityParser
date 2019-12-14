from shapely.geometry import Polygon, LineString, LinearRing

a = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
b = LineString([(1, 1), (2, 2)])
c = LineString([(1, 1), (1, -1)])

print(a.touches(b))
print(a.touches(c))
print(a.crosses(b))
print(a.crosses(c))
print(a.intersects(b))
print(a.intersects(c))
