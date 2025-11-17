"""KML generator

The KML class collect KML entities and saves them to a file when the `close
()` method is called. Note that the KML file is not saved if the KML object
is deleted without being closed.

The methods are used to create KML files:

- add_linestyle() creates a linestyle for use by line entities

- add_markerstyle() creates a markerstyle for use by market entities

- add_folder() creates a folder to contain entities

- add_line() creates a line entity

- add_marker() creates a marker entity
"""

class KML:
    """KML generator class"""

    def __init__(self,
        kmlfile:str,
        ):
        """Start KML file"""

        self.kmlfile = kmlfile
        self.lines = {}
        self.linestyle = {}
        self.marker = {}
        self.markerstyle = {}
        self.folders = {}

    def add_linestyle(self,name:str,**kwargs):
        """Add a line style

        Arguments:

        name: linestyle name

        color: line color

        width: line width

        opacity: line opacity
        """
        self.linestyle[name] = kwargs

    def add_markerstyle(self,name:str,**kwargs):
        """Add a marker style

        Arguments:

        name: markerstyle name

        icon: icon URL

        scale: icon size
        """
        self.markerstyle[name] = kwargs

    def add_folder(self,name:str,**kwargs):
        """Add a folder

        Arguments:

        name: folder name

        parent: parent folder name
        """
        self.folder[name] = kwargs

    def add_line(self,name:str,**kwargs):
        """Add a line entity

        Arguments:

        name: line name

        from_position: line starting position

        to_position: line ending position:

        style: line style

        data: line data
        """
        self.line[name] = kwargs

    def add_marker(self,name:str,**kwargs):
        """Add a marker entity

        Arguments:

        name: marker name

        position: marker position

        style: marker style

        data: marker data
        """
        self.marker[name] = kwargs

    def close(self):
        """Close KML file"""
        if self.kmlfile:
            with open(self.kmlfile,"w",encoding="utf-8") as fh:

                print("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" />
<Document>""",file=fh)

                # TODO: output entities

                print("""</Document>""",file=fh)

            self.kmlfile = None

# if __name__ == "__main__":

#     from ppmodel import PPModel
#     from wecc240 import wecc240
#     PPModel("wecc240").set_case(wecc240()).save_kml("tests/wecc240.kml")
