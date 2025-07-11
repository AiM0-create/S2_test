import streamlit as st
from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.session import AWSSession
from rasterio.windows import Window
import numpy as np
import folium
from streamlit_folium import folium_static
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import tempfile

st.title("Sentinel-2 L2A Viewer (Open Data, No Credentials Required)")

# --- 1. User selects area and date ---
lat = st.number_input("Latitude", value=48.8584)
lon = st.number_input("Longitude", value=2.2945)
buffer = st.slider("Buffer (degrees)", 0.01, 0.1, 0.02)
start_date = st.date_input("Start Date", value=None)
end_date = st.date_input("End Date", value=None)

bbox = [lon-buffer, lat-buffer, lon+buffer, lat+buffer]
if start_date and end_date:
    date_range = f"{start_date}/{end_date}"
else:
    date_range = None

# --- 2. Query STAC for Sentinel-2 L2A imagery (Planetary Computer) ---
st.write("Searching for available images...")
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=date_range,
    query={"eo:cloud_cover": {"lt": 30}},
    max_items=5,
)
items = list(search.get_items())
if not items:
    st.warning("No Sentinel-2 images found for this area/dates.")
    st.stop()

# --- 3. User selects the image ---
options = [f"{item.datetime:%Y-%m-%d %H:%M} | Cloud {item.properties.get('eo:cloud_cover', 'NA')}%" for item in items]
sel_idx = st.selectbox("Select an image", range(len(options)), format_func=lambda i: options[i])
item = items[sel_idx]

# --- 4. Band selection ---
bands = st.multiselect(
    "Select Bands (e.g. B04: Red, B03: Green, B02: Blue, B08: NIR)",
    ["B04", "B03", "B02", "B08"],
    default=["B04", "B03", "B02"]
)

# --- 5. Load selected bands as a small preview ---
def read_band(asset_href, bbox, bands):
    signed_href = planetary_computer.sign(asset_href)
    with rasterio.open(signed_href) as src:
        # Convert bbox to pixel window
        left, bottom, right, top = bbox
        row_start, col_start = src.index(left, top)
        row_stop, col_stop = src.index(right, bottom)
        window = Window.from_slices((min(row_start, row_stop), max(row_start, row_stop)), (min(col_start, col_stop), max(col_start, col_stop)))
        arrs = []
        for band in bands:
            # Asset HREF for each band (in the assets dict)
            asset_band_href = planetary_computer.sign(item.assets[band].href)
            with rasterio.open(asset_band_href) as bsrc:
                arr = bsrc.read(1, window=window, out_shape=(256, 256))
                arrs.append(arr)
        arrs = np.stack(arrs, axis=-1)
        return arrs, src.transform

st.write("Loading image preview...")
preview, transform = read_band(item.assets["B04"].href, bbox, bands)

# --- 6. Normalize and display as RGB ---
def normalize(arr):
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 3000) / 3000  # S2 reflectance scaling
    return arr

img_rgb = normalize(preview[..., :3]) if preview.shape[-1] >= 3 else normalize(preview[..., 0])

st.image(img_rgb, caption="Sentinel-2 Preview", use_column_width=True)

# --- 7. Map widget with area shown ---
m = folium.Map(location=[lat, lon], zoom_start=13)
folium.Rectangle([[lat-buffer, lon-buffer], [lat+buffer, lon+buffer]], color="blue", fill=False).add_to(m)
folium.Marker([lat, lon], tooltip="Center").add_to(m)
folium_static(m)

# --- 8. Download GeoTIFF of selected bands ---
if st.button("Download current selection as GeoTIFF"):
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        with rasterio.open(
            tmp.name, "w",
            driver="GTiff",
            height=preview.shape[0],
            width=preview.shape[1],
            count=preview.shape[2],
            dtype=preview.dtype,
            crs="EPSG:4326",
            transform=transform
        ) as dst:
            for i in range(preview.shape[2]):
                dst.write(preview[..., i], i+1)
        st.download_button(
            label="Download GeoTIFF",
            data=open(tmp.name, "rb").read(),
            file_name="sentinel2_selection.tif",
            mime="image/tiff"
        )

st.markdown("""
---
Data via Microsoft Planetary Computer ([sentinel-2-l2a](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)).
Open access. No login required.
""")
