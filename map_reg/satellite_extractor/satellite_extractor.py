import glob
import math
import shutil
import sys
import os
from dataclasses import dataclass, field
import cv2
import numpy as np

sys.path.append(
    f"{os.path.dirname(os.path.realpath(__file__))}/jimutmap/jimutmap")
from jimutmap import api


@dataclass
class ImageTile:
    x_tile_coor: int
    y_tile_coor: int
    img: np.ndarray


@dataclass
class TiledImage:
    x_tiles: int
    y_tiles: int
    one_tile_res: int
    zoom: int = 19
    __out_img: np.ndarray | None = None
    __tiles: list[list[ImageTile | None]] = field(default_factory=list)

    def __post_init__(self):
        self.__tiles = [[None for _ in range(self.y_tiles)] for _ in range(self.x_tiles)]

    def add_tile(self, tile: ImageTile, idx: tuple[int, int]):
        self.__tiles[idx[0]][idx[1]] = tile

    def stitch(self) -> np.ndarray:
        if self.__out_img is None:
            out_horizontal_tiles = []
            empty_tile = np.zeros((self.one_tile_res, self.one_tile_res, 3),
                                  dtype=np.uint8)
            for x_tile in range(self.x_tiles):
                out_vertical_tiles = []
                for y_tile in range(self.y_tiles):
                    if self.__tiles[x_tile][y_tile] is not None:
                        out_vertical_tiles.append(self.__tiles[x_tile][y_tile].img)
                    else:
                        out_vertical_tiles.append(empty_tile)
                out_horizontal_tiles.append(cv2.vconcat(out_vertical_tiles))
            self.__out_img = cv2.hconcat(out_horizontal_tiles)
        return self.__out_img

    def get_lat_lon_from_xy(self, x_coord: int, y_coord: int) -> tuple[float, float]:
        if self.__out_img is None:
            self.stitch()

        if x_coord < 0 or y_coord < 0:
            raise ValueError("x_coord and y_coord must be positive")

        x_tile_coord = x_coord // self.one_tile_res
        y_tile_coord = y_coord // self.one_tile_res

        if x_tile_coord >= self.x_tiles or y_tile_coord >= self.y_tiles:
            raise ValueError("x_coord and y_coord must be within the image")

        tile = self.__tiles[x_tile_coord][y_tile_coord]

        lat_tile_start, lon_tile_start = ret_lat_lon(zoom=self.zoom,
                                                     x_tile=tile.x_tile_coor,
                                                     y_tile=tile.y_tile_coor)
        lat_tile_end, lon_tile_end = ret_lat_lon(zoom=self.zoom,
                                                 x_tile=tile.x_tile_coor + 1,
                                                 y_tile=tile.y_tile_coor + 1)

        x_coord_in_tile = x_coord % self.one_tile_res
        y_coord_in_tile = y_coord % self.one_tile_res

        lat = lat_tile_start + (lat_tile_end - lat_tile_start) * (y_coord_in_tile / self.one_tile_res)
        lon = lon_tile_start + (lon_tile_end - lon_tile_start) * (x_coord_in_tile / self.one_tile_res)

        return lat, lon


def load_tiles(path_to_imgs: list[str]) -> list[ImageTile]:
    tiles = []
    for img_path in path_to_imgs:
        x_tile = int(img_path.split('/')[-1].split('_')[0])
        y_tile = int(img_path.split('/')[-1].split('_')[1].split('.')[0])
        tiles.append(ImageTile(x_tile_coor=x_tile, y_tile_coor=y_tile,
                               img=np.array(cv2.imread(img_path))))
    return tiles


def stitch_tiles(tiles: list[ImageTile]) -> TiledImage:
    min_x_tile = min([x.x_tile_coor for x in tiles])
    min_y_tile = min([x.y_tile_coor for x in tiles])

    max_x_tile = max([x.x_tile_coor for x in tiles])
    max_y_tile = max([x.y_tile_coor for x in tiles])

    out_tile = TiledImage(x_tiles=max_x_tile - min_x_tile + 1,
                          y_tiles=max_y_tile - min_y_tile + 1,
                          one_tile_res=tiles[0].img.shape[0])

    for tile in tiles:
        out_tile.add_tile(tile=tile, idx=(tile.x_tile_coor - min_x_tile, tile.y_tile_coor - min_y_tile))

    out_tile.stitch()
    return out_tile


# Author Jimut Bahan Pal | jimutbahanpal@yahoo.com
# Copied from jimutmap/jimutmap/jimutmap.py
def ret_lat_lon(zoom: int, x_tile: int, y_tile: int) -> tuple[float, float]:
    n = 2 ** zoom
    lon_deg = int(x_tile) / n * 360.0 - 180.0
    #         lat_rad = math.atan(math.asinh(math.pi * (1 - 2 * int(yTile)/n)))
    lat_rad = 2 * ((math.pi / 4) - math.atan(
        math.exp(-1 * math.pi * (1 - 2 * int(y_tile) / n))))
    lat_deg = lat_rad * 180.0 / math.pi
    return lat_deg, lon_deg


class SatelliteExtractor:

    def __init__(self, path_to_folder: str = "sat_imgs"):
        self.path_to_folder = path_to_folder
        self.zoom = 19

    def download_imgs(self, lat: float, lon: float, radius: float,
                      stitch: bool = True) -> TiledImage | None:
        download_obj = api(min_lat_deg=lat - radius,
                           max_lat_deg=lat + radius,
                           min_lon_deg=lon - radius,
                           max_lon_deg=lon + radius,
                           zoom=self.zoom,
                           verbose=False,
                           threads_=12,
                           container_dir=self.path_to_folder)

        download_obj.download(getMasks=False)
        self.clean_chrome_tmp_folder()

        if stitch:
            downloaded_imgs = download_obj.current_imgs
            if len(downloaded_imgs) == 0:
                print("No images were downloaded, retrying...")
                return self.download_imgs(lat=lat, lon=lon, radius=radius,
                                          stitch=stitch)
            tiles = load_tiles(path_to_imgs=downloaded_imgs)
            return stitch_tiles(tiles=tiles)

    def download_img_range(self, lat: float, lon: float, x_range: int,
                           y_range: int, stitch: bool = True) -> TiledImage | None:
        download_obj = api(min_lat_deg=lat - y_range,
                           max_lat_deg=lat + y_range,
                           min_lon_deg=lon - x_range,
                           max_lon_deg=lon + x_range,
                           zoom=self.zoom,
                           verbose=False,
                           threads_=12,
                           container_dir=self.path_to_folder)

        download_obj.download_range(centre_lat=lat, centre_lon=lon,
                                    x_range=x_range, y_range=y_range)
        self.clean_chrome_tmp_folder()

        if stitch:
            downloaded_imgs = download_obj.current_imgs
            if len(downloaded_imgs) == 0:
                raise ValueError("No images were downloaded")
            tiles = load_tiles(path_to_imgs=downloaded_imgs)
            return stitch_tiles(tiles=tiles)

    # Author Jimut Bahan Pal | jimutbahanpal@yahoo.com
    # Copied from jimutmap/jimutmap/jimutmap.py
    def clean_chrome_tmp_folder(self):
        chromedriver_folders = glob.glob('[0-9]*')
        for item in chromedriver_folders:
            shutil.rmtree(item)


# It expects tiled_img to be a TiledImage object and as param[0]
def click_on_img_lat_lon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        tiled_img = param[0]
        lat, lon = tiled_img.get_lat_lon_from_xy(x, y)
        print(f"LatLon: {lat}, {lon}")


if __name__ == '__main__':
    sat_extractor = SatelliteExtractor()
    out_img = sat_extractor.download_img_range(lat=48.37017393185813,
                                               lon=17.494309514887174,
                                               x_range=2, y_range=2,
                                               stitch=True)

    img = out_img.stitch()
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    cv2.setMouseCallback("img", click_on_img_lat_lon, [out_img])

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
