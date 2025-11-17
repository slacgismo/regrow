"""KML generator"""

class KML:
    """KML generator class"""

    def __init__(self,
        kmlfile:str,
        ):
        """Start KML file"""

        self.kmlfile = kmlfile

    def __del__(self):
        """Cleanup"""
        if self.kmlfile:
            self.close()

    def set_linestyle(self,**kwargs):

        pass

    def set_markerstyle(self,**kwargs):

        pass

    def add_line(self,**kwargs):

        pass

    def add_marker(self,**kwargs):

        pass

    def close(self):
        """Close KML file"""
        if self.kmlfile:
            with open(self.kmlfile,"w",encoding="utf-8") as fh:

                print("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" />
<Document>""",file=fh)

                print("""</Document>""",file=fh)

            self.kmlfile = None

if __name__ == "__main__":

    from ppmodel import PPModel
    from wecc240 import wecc240
    PPModel("wecc240").set_case(wecc240()).to_kml("tests/wecc240.kml")
