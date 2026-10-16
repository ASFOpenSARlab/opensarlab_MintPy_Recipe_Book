import ipywidgets as widgets
from ipyleaflet import Map, DrawControl, basemaps, WidgetControl, basemap_to_tiles, TileLayer, GeoJSON, Rectangle
from pyproj import Transformer

def bbox_to_wkt(bounds):
    """
    Convert ipyleaflet bounds to WKT.

    WKT uses:
        lon lat
    """
    (south, west), (north, east) = bounds

    return (
        "POLYGON(("
        f"{west} {south}, "
        f"{east} {south}, "
        f"{east} {north}, "
        f"{west} {north}, "
        f"{west} {south}"
        "))"
    )

def rectangular_aoi_select(gunw_bbox):
    """
    Generate a global map from which a user can select a rectangular area of interest.

    Returns: A dictionary containing a Map object and a Textarea widget containing the WKT of a drawn rectangle 
    """
    
    west, south, east, north = gunw_bbox

    bounds = (
        (south, west),
        (north, east),
    )
    x_pad = (east - west) * 0.1
    y_pad = (north - south) * 0.1
    
    padded_bounds = (
        (south - y_pad, west - x_pad),
        (north + y_pad, east + x_pad),
    )

    center = (
        (south + north) / 2,
        (west + east) / 2,
    )
    
    m = Map(
        center=center,
        zoom=8,
        basemap={},
        scroll_wheel_zoom=True,
        layout=widgets.Layout(width='800px', height='800px')
    )

    
    m.add(basemap_to_tiles(basemaps.Esri.WorldImagery))
    labels = TileLayer(
        url="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attribution="Esri",
        name="Esri Boundaries and Places",
    )

    m.add(
        Rectangle(
            bounds=(bounds),
            color="red",
            weight=2,
            fill_opacity=0.0,
        )
    )
    
    legend = widgets.HTML('''
    <span style="color:red">━━</span> GUNW Raster Bounds
    </br>
    <span style="color:orange">━━</span> AOI Bounds
    ''')
    m.add_control(WidgetControl(widget=legend, position="topright"))
    
    m.add(labels)
    
    wkt_box = widgets.Textarea(
        value="Draw a rectangle on the map...",
        description="AOI WKT:",
        layout=widgets.Layout(width="500px", height="90px"),
    )
    
    draw_control = DrawControl(
        rectangle={
            "shapeOptions": {
                "color": "orange",
                "weight": 3,
                "fillOpacity": 0.15,
            }
        },
        polygon={},
        circlemarker={},
        polyline={},
        marker={},
        circle={},
        edit=False,
        remove=False,
    )
    
    last_rectangle = {"feature": None}

    selected_rectangle = GeoJSON(data={"type": "FeatureCollection", "features": []})
    m.add(selected_rectangle)

    
    def handle_draw(target, action, geo_json):
        if action != "created":
            return
    
        # copy the new rectangle into our persistent layer
        selected_rectangle.data = {
            "type": "FeatureCollection",
            "features": [geo_json],
        }
    
        # clear the DrawControl layer
        draw_control.clear()
    
        coords = geo_json["geometry"]["coordinates"][0]
    
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
    
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
    
        wkt_box.value = (
            "POLYGON(("
            f"{west} {south}, "
            f"{east} {south}, "
            f"{east} {north}, "
            f"{west} {north}, "
            f"{west} {south}"
            "))"
        )
    draw_control.on_draw(handle_draw)

    m.add_control(draw_control)
    m.add_control(WidgetControl(widget=wkt_box, position="bottomleft"))
    return {
        "map": m,
        "wkt_box": wkt_box,
    }

def rectangular_wkt_bounds(wkt: str) -> dict[str, float]:
    coords_text = (
        wkt.strip()
        .removeprefix("POLYGON((")
        .removesuffix("))")
    )

    coords = []
    for pair in coords_text.split(","):
        lon, lat = map(float, pair.strip().split())
        coords.append((lon, lat))

    lons = [lon for lon, lat in coords]
    lats = [lat for lon, lat in coords]

    return {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats),
    }