"""Convert db files to kml files for review"""
import pandas as pd
import utils

pd.options.display.width=None
pd.options.display.max_columns=None

gis = pd.read_csv("wecc240_gis.csv")
gis["bus"] = [utils.geohash(x,y) for x,y in zip(gis.Lat,gis.Long)]
gis.set_index("bus",inplace=True)

for file in ["uspvdb.csv","uswtdb.csv"]:
    data = gis.join(pd.read_csv(file,index_col=["bus"]))

    with open(file.replace(".csv",".kml"),"w") as kml:

        print("""<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
    """,file=kml)
        print(f"""  <Document>
        <name>{file}</name>
        <Style id="pv">
          <IconStyle>
            <colorMode>normal</colorMode>
            <scale>1</scale>
            <heading>0</heading>
            <Icon>
                <href>http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_yellow.png</href>
            </Icon>
          </IconStyle>
        </Style>
        <Style id="link">
            <LineStyle>
              <color>ff00ffff</color>
              <width>1</width>
            </LineStyle>
            <PolyStyle>
              <color>ff00ffff</color>
            </PolyStyle>
          </Style>""",file=kml)

        for bus,item in data.iterrows():

            print(f"""    <Placemark>
          <name>{str(item['name']).replace('&','')}</name>
          <styleUrl>#pv</styleUrl>
          <Point><coordinates>{item.longitude},{item.latitude}</coordinates></Point>
        </Placemark>
        <Placemark>
          <styleUrl>#link</styleUrl>
          <LineString>
            <coordinates>
                {item.longitude},{item.latitude},50
                {item.Long},{item.Lat},50
            </coordinates>
          </LineString>
        </Placemark>""",file=kml)

        print("""  </Document>""",file=kml)

        print("""</kml>""",file=kml)

