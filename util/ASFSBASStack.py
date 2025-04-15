from datetime import datetime, timedelta
import re

import asf_search as asf
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape, Polygon
from tqdm import tqdm

tqdm.pandas()

"""
    TODOs:
      - support add/remove pairs
      - optionally connect both beginning and end of each season with 1-year pairs (currently connects ends only)
      - optional arg for target date from which to create bridge pairs
      - optional automatic disconnected-stack correction 
        - If there is an interruption in data at the end of a season, a disconnection can occur
          - work backwards from the end of the season until a scene is found that can connect to the following year
        - if the final season in a stack contains no scene within the temporal baseline threshold of the end of the season, it 
          will be disconnected
          - solution: perform backwards search to connect end of stack to previous year (maintain earlier date as ref scene)
"""


class ASFSBASStack:
    """
    The ASFSBASStack class creates connected SBAS stacks from Sentinel-1 SLCs and Sentinel-1 SLC bursts. All pairs are created based
    on the currently set temporal and perpendicular baselines. If a season is defined, out-of-season acquisitions are filtered and
    seasonal gaps are bridged with 1-year pairs. To determine the number of seasonal bridge pairs to create, a quantity is estimated
    by dividing the temporal baseline by an expected repeat pass frequency for the satellite (12 days for Sentinel-1). If a
    reference scene is expected to have 3 pairs, and one is available in-season, only 2 out-of-season pairs will be created.

    Stacks are not guaranteed to be connected. A lack of data acquired at a given time and place can cause gaps.

    Performs lazy evaluation when updating stack parameters. SBAS stack pairs are not recalculated until self.sbas_stack is accessed.

    Key methods:
        - `plot()`
        - `get_insar_pairs()`

    """

    def __init__(self, **kwargs: dict):
        self.plot_available = False
        try:
            import plotly.graph_objects as go
            import networkx as nx

            self.plot_available = True
        except ImportError:
            print(
                "Warning: ASFSBASStack.plot() requires additional dependencies: plotly and/or networkx"
            )

        self._geo_ref_scene_id = kwargs.get("refSceneName", None)
        self._burst_ids = kwargs.get("burstIDs", None)
        self._burst_swaths = kwargs.get("burstSwaths", None)
        self._season = kwargs.get("season", ("1-1", "12-31"))
        self._start = kwargs.get("start", None)
        self._end = kwargs.get("end", None)
        self.needs_sbas_stack_update = True
        self.no_auto_stack_update = False
        self._sbas_stack = None
        self._is_multi_burst = False

        self.bridge_target_date = kwargs.get("bridgeTargetDate", self._calc_midseason_date())
        self.perp_baseline = kwargs.get("perpendicularBaseline", 400)
        self.temporal_baseline = kwargs.get("temporalBaseline", 36)
        self.repeat_pass_freq = kwargs.get("repeatPassFrequency", 12)
        self.overlap_threshold = kwargs.get("overlapThreshold", 0.8)

        # build stack upon initialization if min required args were passed
        if self._geo_ref_scene_id and self._end:
            self._sbas_stack = self.build_sbas_stack()
            # self.needs_sbas_stack_update = False

        if not self._geo_ref_scene_id and self._burst_ids and self._burst_swaths:
            self._is_multi_burst = True
            opts = {
                'platform': asf.PLATFORM.SENTINEL1,
                'fullBurstID': f"{self._burst_ids[0]}_{self._burst_swaths[0]}",
                'season': self._get_jullian_season(),
                'start': pd.to_datetime(self._start, utc=True),
                'end': pd.to_datetime(self._end, utc=True),
                'polarization': 'VV'
            }
            results = asf.geo_search(intersectsWith=None, **opts)
            
            self._geo_ref_scene_id = sorted([r.properties["sceneName"] for r in results])[0]
            self._multi_burst_stacks = list()
            if self._end:
                self._sbas_stack = self.build_sbas_stack()
                self.get_multi_bursts()
                self.no_auto_stack_update = True
                self.multi_burst_pairs = self._get_multi_burst_pairs()
                self.sync_multi_burst_stacks()

    @property
    def full_stack(self):
        return self._full_stack

    @property
    def geo_ref_scene_id(self):
        return self._geo_ref_scene_id

    @geo_ref_scene_id.setter
    def geo_ref_scene_id(self, scene_id: str):
        if self._is_multi-burst:
            raise Exception(
                "You cannot set the geographic reference scene on a multi-burst stack"
            )
        self._geo_ref_scene_id = scene_id
        self._sbas_stack = None
        self.needs_sbas_stack_update = True

    @property
    def multi_burst_stacks(self):
        return self._multi_burst_stacks

    @property
    def burst_ids(self):
        return self._burst_ids

    @property
    def burst_swaths(self):
        return self._burst_swaths

    @property
    def season(self):
        return self._season

    @season.setter
    def season(self, season: tuple[str, str]):
        self._season = season
        self.needs_sbas_stack_update = True

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, stack_start_date: str):
        if hasattr(self, "_sbas_stack") and not (
            pd.to_datetime(self._sbas_stack["stopTime"]).min()
            <= pd.to_datetime(stack_start_date, utc=True)
            <= pd.to_datetime(self._sbas_stack["stopTime"]).max()
        ):
            self.needs_sbas_stack_update = True
        self._start = stack_start_date

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, stack_end_date: str):
        if hasattr(self, "_sbas_stack") and not (
            pd.to_datetime(self._sbas_stack["stopTime"]).min()
            <= pd.to_datetime(stack_end_date, utc=True)
            <= pd.to_datetime(self._sbas_stack["stopTime"]).max()
        ):
            self.needs_sbas_stack_update = True
        self._end = stack_end_date

    @property
    def sbas_stack(self):
        if self.needs_sbas_stack_update:
            self._sbas_stack = self.build_sbas_stack()
            self._full_stack = self._build_full_stack()
            self.needs_sbas_stack_update = False
        elif not self.no_auto_stack_update:
            self._sbas_stack["insarNeighbors"] = self._sbas_stack.apply(
                lambda row: self._get_seasonal_nearest_neighbors(
                    self._sbas_stack, row["sceneName"]
                ),
                axis=1,
            )
        return self._sbas_stack

    @sbas_stack.setter
    def sbas_stack(self, stack):
        self.needs_sbas_stack_update = False
        self._sbas_stack = stack

    @property
    def plot(self):
        if not self.plot_available:
            raise Exception(
                "ASFSBASStack.plot() requires additional dependencies: plotly and/or networkx"
            )
        return self._plot

    def _build_full_stack(self):
        args = asf.ASFSearchOptions(
            **{"start": self._start, "end": self._end}
        )
    
        # get baseline stack from stack ref scene
        search_scene_id = self._geo_ref_scene_id if 'BURST' in self._geo_ref_scene_id else f'{self._geo_ref_scene_id}-SLC'
        stack = asf.stack_from_id(search_scene_id, args)

        # put the stack in a GeoDataFrame
        return gpd.GeoDataFrame(
            [s.properties | s.baseline for s in stack],
            geometry=[shape(s.geometry) for s in stack],
        )

    
    def _calc_midseason_date(self):
        season_start = datetime.strptime(self.season[0], "%m-%d").timetuple().tm_yday
        season_end = datetime.strptime(self.season[1], "%m-%d").timetuple().tm_yday
        
        if season_start < season_end:
            midseason = ((season_end - season_start) // 2) + season_start
        elif season_end < season_start:
            half_length = (365 - season_start + season_end) // 2
            if half_length <= 365 - season_start:
                midseason = season_start + half_length
            else:
                midseason = half_length - (365 - season_start)
        else:
            raise Exception("Cannot calculate a midseason date for a season of 0 length")

        return (datetime(2000, 1, 1) + timedelta(days=midseason - 1)).strftime("%m-%d")
        

    def _merge_stacks(self, stack_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Merges the passed in SBAS stack GeoDataFrame with self.sbas_stack. Any previously existing reference
        scene stack search results stored in the 'stack' column of sekf.sbas_stack are preserved. This allows the
        collection of stack searches to be grown as changes to the time bounds of the stack or seasons warrant,
        preventing duplicated stack searches.

        Arguments:
            stack_gdf (gpd.GeoDataFrame): A new GeoDataFrame with only self._geo_ref_scene_id's stack search results and
            an empty 'insarNeighbors' column

        Returns: A merged GeoDataFrame
        """
        if self._sbas_stack is not None:
            # merge updated stack_gdf with existing _sbas_stack so we only need to make API calls on added scenes
            stack_gdf = self._sbas_stack.merge(stack_gdf, on=["sceneName"], how="right")

            # merge resulting stack_x and stack_y columns
            stack_gdf["stack"] = stack_gdf["stack_y"].combine_first(
                stack_gdf["stack_x"]
            )

            # copy appropriate _x and _y columns to columns with their original names
            x_to_copy = ["insarNeighbors_x"]
            y_to_copy = [
                i
                for i in stack_gdf.columns
                if "_y" in i and "insarNeighbors" not in i and "stack" not in i
            ]

            for y in y_to_copy:
                stack_gdf[y.split("_y")[0]] = stack_gdf[y]
            for x in x_to_copy:
                stack_gdf[x.split("_x")[0]] = stack_gdf[x]

            # clean up _x and _y columns
            to_drop = [i for i in stack_gdf.columns if "_x" in i or "_y" in i]
            stack_gdf = stack_gdf.drop(columns=to_drop)

        return stack_gdf

    # def remove_date_pair(self, date_tuple: tuple(str,str)):
        
        

    
    def _centered_sublist(
        self, stack: list[asf.ASFProduct], target: int, length: int
    ) -> list[asf.ASFProduct]:
        """
        Given an asf_search.stack_from_id search results list of ASFProducts and a target temporal baseline, returns a
        sublist of a given length whose temporal baselines are centered around the target.

        Arguments:
            stack  (list[asf_search.ASFProduct]): The results of an asf_search.stack_from_id search
            target (int): The day of the year to center the sublist around
            length (int): The length of the sublist

        Returns:
            A sublist of the given length whose temporal baselines are centered around the target

        """
        temp_baselines = [i.properties["temporalBaseline"] for i in stack]
        if len(temp_baselines) == 0:
            return []
        closest_value = min(temp_baselines, key=lambda x: abs(x - target))
        target_index = temp_baselines.index(closest_value)
        half_length = length // 2

        start_index = max(0, target_index - half_length)
        end_index = min(len(stack), start_index + length)

        if end_index - start_index < length:
            start_index = max(0, end_index - length)

        return [stack[i] for i in range(start_index, end_index)]

    def _baseline_stack_filter(
        self,
        ref_scene_gdf: gpd.geodataframe.GeoDataFrame,
        stack: list[asf.ASFProduct],
        temporal_baseline_range: tuple[int, int],
    ):
        """
        Takes an input asf_search.stack_from_id search results list and a range of temporal baselines. Returns a sublist
        of asf_search.ASFProducts whose temporal baselines fall within the given range, whose perpendicular baseline
        is less than or equal to self.perp_baseline, and that would produce an inSAR pair whose geometry _overlaps with the
        stack reference scene within the set overlap_threshold.

        Arguments:
            stack (list[asf_search.ASFProduct]): asf_search.stack_from_id search results list
            temporal_baseline_range (tuple[int, int]): Range of temporal baselines by which to filter ASFProducts
        Returns:
            A sublist of asf_search.ASFProducts whose temporal baselines fall within the given range and whose
            perpendicular baseline is less than or equal to self.perp_baseline

        """
        ref_scene_geo = ref_scene_gdf.geometry.iloc[0]
        ref_scene_name = ref_scene_gdf.iloc[0]["sceneName"]


        ref_scene_dt = pd.Timestamp(ref_scene_gdf["stopTime"].iloc[0], tz="UTC")
        return [
            i
            for i in stack
            if (
                temporal_baseline_range[0]
                < (pd.Timestamp(i.properties["stopTime"], tz="UTC") - ref_scene_dt).days
                <= temporal_baseline_range[1]                    
                and np.abs(self._calc_perp_baseline(ref_scene_gdf.iloc[0]['perpendicularBaseline'], i.properties["perpendicularBaseline"]))
                <= self.perp_baseline
                and pd.Timestamp(i.properties["stopTime"], tz="UTC")
                <= pd.Timestamp(self._end, tz="UTC")
                and pd.Timestamp(i.properties["stopTime"], tz="UTC") >= ref_scene_dt
                and self._overlaps(ref_scene_geo, shape(i.geometry))
                and i.properties["sceneName"] != ref_scene_name
            )
        ]
            

    def _get_seasonal_nearest_neighbors(
        self, stack_gdf: gpd.GeoDataFrame, ref_scene_name: str
    ) -> list[asf.ASFProduct]:
        """
        Used to recalculate the SBAS stack pairs.

        Applies the perpendicular baseline threshold between the reference and secondary scenes in each InSAR pair.

        Given an sbas_stack GeoDataFrame with a complete 'stack' column and a reference scene ID that it contains
        (in the 'sceneName' column), returns a list of the secondary inSAR scenes for the given reference scene.
        If the expected number of pairs cannot be made within the current season, 1-year pairs (+ or - the
        temporal baseline) are attempted. The expected number of pairs for each reference scene is determined by
        `self.temporal_baseline // self.repeat_pass_freq`

        Arguments:
            stack_gdf (geopandas.GeoDataFrame): GeoDataFrame with a complete 'stack' column
            ref_scene_name (str): ID of the reference scene for whose secondary scenes we are searching

        Returns:
            A list of secondary scenes for the provided reference scene that adheres to the currently defined
            contraints of the SBAS stack

        """

        stack = stack_gdf.loc[stack_gdf.sceneName == self._geo_ref_scene_id][
            "stack"
        ].iloc[0]

        # create stack of in-season scenes, within baseline thresholds
        cur_season_stack = self._baseline_stack_filter(
            stack_gdf.loc[stack_gdf.sceneName == ref_scene_name],
            stack,
            (0, self.temporal_baseline),
        )

        # find the number of expected neighbors
        neighbor_len = self.temporal_baseline // self.repeat_pass_freq

        # return the current season's stack if enough neighbors were found
        ref_scene_date = stack_gdf.loc[stack_gdf.sceneName == ref_scene_name]['stopTime'].iloc[0]
        ref_scene_date = pd.to_datetime(ref_scene_date, utc=True)
        bridge_target_date = pd.to_datetime(f"{ref_scene_date.year}-{self.bridge_target_date}", utc=True)
        bridge_range_start = bridge_target_date - pd.Timedelta(days=self.temporal_baseline)
        bridge_range_end = bridge_target_date + pd.Timedelta(days=self.temporal_baseline)
 
        if (self._season == ('1-1', '12-31')
            or ref_scene_date < bridge_range_start
            or ref_scene_date > bridge_range_end):
            
            return cur_season_stack

        # create stack of next-season scenes
        next_season_stack = self._baseline_stack_filter(
            stack_gdf.loc[stack_gdf.sceneName == ref_scene_name],
            stack,
            (365 - self.temporal_baseline, 365 + self.temporal_baseline),
        )

        # # find number remaining neighbors to identify
        # neighbor_len = neighbor_len - len(cur_season_stack)

        # return any current season and next season scenes found
        return cur_season_stack + self._centered_sublist(
            next_season_stack, 365, neighbor_len
        )

    def get_insar_pairs(self) -> list[asf.ASFProduct]:
        """
        Useful when ordering an SBAS stack from ASF HyP3 On-Demand Processing

        If an ASFSBASStack.sbas_stack GeoDataFrame is available, generate a list tuples containing the
        reference and secondary scene name for each inSAR pair in the SBAS stack.

        Returns:
            A list tuples containing the reference and secondary scene name for each inSAR pair in the SBAS stack
        """

        return [
            (row["sceneName"], neighbor.properties["sceneName"])
            for _, row in self.sbas_stack.iterrows()
            for neighbor in row["insarNeighbors"]
        ]

    def get_multi_bursts(self):
        ts_regex = r"(?<=IW1_|IW2_|IW3_)\d{8}T\d{6}"
        start = re.search(ts_regex, self._geo_ref_scene_id).group(0)
        start = pd.Timestamp(start, tz="UTC")
        end = start + pd.Timedelta(hours=1)
        self._multi_burst_stacks = list()
        for id in self._burst_ids:
            opts = {
                'platform': asf.PLATFORM.SENTINEL1,
                'relativeBurstID': id.split("_")[1],
                'start': start,
                'end': end,
                'polarization': 'VV'
            }
            results = asf.geo_search(intersectsWith=None, **opts)
            scene_ids = [r.properties["sceneName"] for r in results]
            scene_ids = [i for i in scene_ids if i != self._geo_ref_scene_id and i.split("_")[2] in self._burst_swaths]
            for id in scene_ids:
                args = {
                    'refSceneName': id,
                    'season': self._season, 
                    'start': self._start, 
                    'end': self._end, 
                    'perpendicularBaseline': self.perp_baseline, 
                    'temporalBaseline': self.temporal_baseline,
                    'repeatPassFrequency': self.repeat_pass_freq,
                    'overlapThreshold': self.overlap_threshold,
                }
                self._multi_burst_stacks.append(ASFSBASStack(**args))        

    def _get_multi_burst_pairs(self):
        burst_stack_1_pairs = self.get_insar_pairs()

        stack_by_dates_dict = { 
            (i[0].split('_')[3][:8], i[1].split('_')[3][:8]): [[i[0]],[i[1]]] 
            for i in burst_stack_1_pairs 
        }
        
        for stack in list(self.multi_burst_stacks):
            stack_pairs = stack.get_insar_pairs()
            for pair in stack_pairs:
                date_key = (pair[0].split('_')[3][:8], pair[1].split('_')[3][:8])
                if date_key in stack_by_dates_dict:
                    stack_by_dates_dict[date_key][0].append(pair[0])
                    stack_by_dates_dict[date_key][1].append(pair[1])

        # Remove any pairs that did not have full coverage due
        # to 1+ bursts having too long a perpendicular baseline
        burst_count = len(list(self.multi_burst_stacks)) + 1
        stack_by_dates_dict = {
            k: v for k, v in stack_by_dates_dict.items()
            if len(v[0]) == burst_count and len(v[1]) == burst_count
        }
        return stack_by_dates_dict

    def sync_multi_burst_stacks(self):
        valid_date_pairs = self.multi_burst_pairs.keys()
        for i, row in self.sbas_stack.iterrows():
            new_neighbors = []
            for neighbor in row["insarNeighbors"]:
                date_pair = (
                "".join(row["stopTime"].split("T")[0].split("-")),
                "".join(neighbor.properties["stopTime"].split("T")[0].split("-"))
                )
                if date_pair in valid_date_pairs:
                    new_neighbors.append(neighbor)
        
            self.sbas_stack.at[i, "insarNeighbors"] = new_neighbors

    def _overlaps(self, ref_scene_geo: Polygon, sec_scene_geo: Polygon) -> bool:
        """
        Determines if two scenes overlap enough to satisfy the currently
        set self.overlap_threshold (0-1)

        Arguments:
            ref_scene_geo (shapely.geometry.Polygon): geometry of reference scene
            sec_scene_geo (shapely.geometry.Polygon): geometry of secondary scene

        Returns: True if self.overlap_threshold is met, else False
        """
        if self.needs_sbas_stack_update:
            self._sbas_stack
        intersection_area = ref_scene_geo.intersection(sec_scene_geo).area
        ref_area = ref_scene_geo.area
        overlap = (intersection_area / ref_area) if ref_area != 0 else 0
        return overlap >= self.overlap_threshold

    def _check_if_secondary_scene(self, scene_name: str) -> bool:
        """
        Checks if the passed in scene is present in the 'insarNeighbors' column of self.sbas_stack

        Arguments:
            scene_name (str): The ID of the scene to check

        Returns:
            True if the scene is found in the 'insarNeighbors' column of self.sbas_stack, else False
        
        """
        # for _, row in self.sbas_stack.iterrows():
        #     for i in row["insarNeighbors"]:
        #         if i.properties["sceneName"] == scene_name:
        #             return True
        # return False
        secondary_scenes = [i.properties['sceneName'] for j in self.sbas_stack['insarNeighbors'] for i in j]
        return  scene_name in secondary_scenes
        

    def ref_stack_len(self):
        counter = 0
        for _, row in self.sbas_stack.iterrows():
            if len(row["insarNeighbors"]) > 0 or self._check_if_secondary_scene(
                row["sceneName"]
            ):
                counter += 1
        return counter

    
    def _calc_perp_baseline(self, ref_perp_baseline, sec_perp_baseline):
        if not ref_perp_baseline or not sec_perp_baseline:
            # If a baseline is nan, return a large enough value to flag removal of the pair
            return 1000000
        if ref_perp_baseline * sec_perp_baseline >= 0:
            return sec_perp_baseline - ref_perp_baseline
        elif ref_perp_baseline <= sec_perp_baseline:
            return np.abs(sec_perp_baseline) + np.abs(ref_perp_baseline)
        else:
            return (np.abs(sec_perp_baseline) + np.abs(ref_perp_baseline)) * -1
            
        

    def _plot(self):
        """
        Plot the SBAS stack

        """
        import plotly.graph_objects as go
        import networkx as nx

        import time # TODO remove

        section1_start = time.time()

        G = nx.DiGraph()

        insar_node_pairs = [
            (
                row["stopTime"].split("T")[0],
                neighbor.properties["stopTime"].split("T")[0],
                {
                    "perp_bs": self._calc_perp_baseline(row["perpendicularBaseline"], neighbor.properties["perpendicularBaseline"]),
                    "temp_bs": (
                        pd.Timestamp(neighbor.properties["stopTime"], tz="UTC")
                        - pd.Timestamp(row["stopTime"], tz="UTC")
                    ).days,
                },
            )
            for _, row in self.sbas_stack.iterrows()
            for neighbor in row["insarNeighbors"]
        ]

        G.add_edges_from(insar_node_pairs, data=True)

        scene_dates = {
            row["stopTime"].split("T")[0]: row["stopTime"].split("T")[0]
            for _, row in self._sbas_stack.iterrows()
        }
        nx.set_node_attributes(G, scene_dates, "date")

        perp_bs = {
            row["stopTime"].split("T")[0]: row["perpendicularBaseline"]
            for _, row in self._sbas_stack.iterrows()
        }
        nx.set_node_attributes(G, perp_bs, "perp_bs")

        node_positions = {
            node: datetime.strptime(data["date"], "%Y-%m-%d").timestamp()
            for node, data in G.nodes(data=True)
        }
        node_y_positions = {node: data["perp_bs"] for node, data in G.nodes(data=True)}

        node_x = [node_positions[node] for node in G.nodes()]
        node_y = [node_y_positions[node] for node in G.nodes()]
        node_text = [G.nodes[node]["date"] for node in G.nodes()]

        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0 = node_positions[edge[0]]
            y0 = node_y_positions[edge[0]]
            x1 = node_positions[edge[1]]
            y1 = node_y_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(
                f"{edge[0]} - {edge[1]}, perp baseline: {edge[2]['perp_bs']}, temp baseline: {edge[2]['temp_bs']}"
            )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=4, color="rgba(52, 114, 168, 0.7)"),
            mode="lines",
        )

        edge_hover_trace = go.Scatter(
            x=[
                (node_positions[edge[0]] + node_positions[edge[1]]) / 2
                for edge in G.edges()
            ],
            y=[
                (node_y_positions[edge[0]] + node_y_positions[edge[1]]) / 2
                for edge in G.edges()
            ],
            mode="markers",
            marker=dict(size=20, color="rgba(255, 255, 255, 0)"),
            hoverinfo="text",
            text=edge_text,
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            textposition="top center",
            marker=dict(size=15, color="rgba(32, 33, 32, 0.7)", line_width=0),
            hoverinfo="text",
            hovertext=node_text,
        )

        # create a gdf containing all unused SLCs in the full time range of the time series
        empty_neighbors = self.sbas_stack[self.sbas_stack['insarNeighbors'].apply(lambda x: len(x) > 0)]
        gdf = self._full_stack[~self._full_stack['sceneName'].isin(empty_neighbors['sceneName'])]

        secondary_scenes = [i.properties['sceneName'] for j in self.sbas_stack['insarNeighbors'] for i in j]      
        gdf = gdf[gdf['sceneName'].apply(lambda x: x not in secondary_scenes)]
        
        all_slcs = [i.split("T")[0] for i in gdf["stopTime"]]
        all_slcs_x = [datetime.strptime(i.split("T")[0], "%Y-%m-%d").timestamp() for i in gdf["stopTime"]]  # datetime.strptime(data["date"], "%Y-%m-%d").timestamp()
        all_slcs_y = [i for i in gdf["perpendicularBaseline"]]


        all_slcs_trace = go.Scatter(
            x=all_slcs_x, y=all_slcs_y,
            mode='markers',
            hoverinfo='text',
            text=all_slcs,
            marker=dict(
                color='rgba(255,0,0,0.5)',  # Different color for unconnected nodes
                size=10,
                line_width=2
            )
        )

        date_range = (
            pd.date_range(
                start='-'.join(self.start.split('-')[:2]), 
                end='-'.join(self.end.split('-')[:2]), 
                freq="MS"
            )
            .strftime("%Y-%m")
            .tolist()
        )

        
        date_range_ts = [
            datetime.strptime(date, "%Y-%m").timestamp() for date in date_range
        ]
        
        def f_date(dash_date_str):
            return dash_date_str.replace("-", "/")

        fig = go.Figure(
            data=[edge_trace, edge_hover_trace, node_trace, all_slcs_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                height=800,
                margin=dict(t=180),
                xaxis=dict(
                    title="Acquisition Date",
                    tickvals=date_range_ts,
                    ticktext=date_range,
                    gridcolor="gray",
                    zerolinecolor="gray",
                ),
                yaxis=dict(
                    title="Perpendicular Baseline (m)",
                    gridcolor="gray",
                    zerolinecolor="gray",
                ),
                title=dict(
                    text=(
                        "<b>Sentinel-1 Seasonal SBAS Stack</b><br>"
                        f"Reference: {self._geo_ref_scene_id}<br>"
                        f"Temporal Bounds: {f_date(self._start)} - {f_date(self._end)}, Seasonal Bounds: {f_date(self._season[0])} - {f_date(self._season[1])}<br>"
                        f"Max Temporal Baseline: {self.temporal_baseline} days, Max Perpendicular Baseline: {self.perp_baseline}m<br>"
                        f"Stack Size: {len(insar_node_pairs)} pairs from {self.ref_stack_len()} scenes<br>"
                    ),
                    y=0.95,
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                    font=dict(
                        family="Helvetica, monospace",
                        size=22,
                    ),
                ),
                plot_bgcolor="white",
                paper_bgcolor="lightgrey",
            ),
        )
        fig.show()

    def _get_jullian_season(self) -> tuple[int,int]:
        season_start_ts = pd.Timestamp(
            datetime.strptime(self._season[0], "%m-%d"), tz="UTC"
        )
        season_start_day = season_start_ts.timetuple().tm_yday
        season_end_ts = pd.Timestamp(
            datetime.strptime(self._season[1], "%m-%d"), tz="UTC"
        )
        season_end_day = season_end_ts.timetuple().tm_yday
        return (season_start_day, season_end_day)

    def build_sbas_stack(self) -> gpd.GeoDataFrame:
        """
        Builds or updates the ASFSBASStack.sbas_stack GeoDataFrame based on currently set stack parameters.

        Returns:
            The new or updated SBAS stack GeoDataFrame
        """
        if not self._geo_ref_scene_id:
            raise (Exception('Stack "refSceneName" not set'))
        if not self._start:
            raise (Exception('Stack "start" not set'))
        if not self._end:
            raise (Exception('Stack "end" not set'))
        if pd.to_datetime(self._end) <= pd.to_datetime(self._start):
            raise (
                Exception(
                    "Stack end date is earlier than its start date. Correct to proceed."
                )
            )

        season = self._get_jullian_season()

        args = asf.ASFSearchOptions(
            **{"start": self._start, "end": self._end, "season": season}
        )

        # get baseline stack from stack ref scene
        search_scene_id = self._geo_ref_scene_id if 'BURST' in self._geo_ref_scene_id else f'{self._geo_ref_scene_id}-SLC'

        stack = asf.stack_from_id(search_scene_id, args)

        # put the stack in a GeoDataFrame
        gdf = gpd.GeoDataFrame(
            [s.properties | s.baseline for s in stack],
            geometry=[shape(s.geometry) for s in stack],
        )

        # add 'stack' and 'insarNeighbors' columns
        gdf["insarNeighbors"] = float("nan")
        gdf["stack"] = float("nan")

        # Add ref scene's stack to its row's 'stack' column
        gdf["stack"] = gdf["stack"].astype(object)
        ref_index = gdf.index[gdf["sceneName"] == self._geo_ref_scene_id]
        gdf.at[ref_index[0], "stack"] = stack

        # merge gdf with self.sbas_stack, if one exists
        gdf = self._merge_stacks(gdf)

        # Find neighbors for every potential InSAR reference scene in the stack
        gdf["insarNeighbors"] = gdf.apply(
            lambda row: self._get_seasonal_nearest_neighbors(gdf, row["sceneName"]),
            axis=1,
        )

        return gdf
