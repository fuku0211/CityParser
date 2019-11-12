import shapefile
with shapefile.Reader("data\\shp\\test\\tatemono.shp") as f:

    shp = f.shapes()
    shprecs = f.shapeRecords()
    a = [i for i in shp if len(i.parts)]
    print()