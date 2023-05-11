# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import json
import os
import datetime

from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import pandas as pd
import geopandas as gpd
import folium

from geopy import distance as geo_distance


# -

# # Load Sectors

def read_sector_from_folder(folder: str):
    subsector_list = []
    for path in os.listdir(folder):
        if not path.endswith('.txt'):
            continue
        with open(folder+'/'+path) as f:
            sec_txt = f.read()
        str_cords = sec_txt.split(" - ")
        cords = []
        for c in str_cords:
            lat = int(c[0:2]) + int(c[2:4])/60 + int(c[4:6])/3600
            lon = int(c[8:11]) + int(c[11:13])/60 + int(c[13:15])/3600
            cords.append((lon,lat))
        subsector_list.append(
            gpd.GeoDataFrame(
                index=[0],
                crs='epsg:4326',
                geometry=[Polygon(cords)]
            )
        )
    assert len(subsector_list) > 0
    if len(subsector_list) == 1:
        return subsector_list[0]
    else:
        return gpd.GeoDataFrame(
            pd.concat(subsector_list, ignore_index=True),
            crs=subsector_list[0].crs
        ).dissolve()


sector_w_esmm = read_sector_from_folder('./raw_data/sectors_info/esmm_acc_sector_W')
sector_y_esmm = read_sector_from_folder('./raw_data/sectors_info/esmm_acc_sector_Y')
sector_6_esmm = read_sector_from_folder('./raw_data/sectors_info/esmm_acc_sector_6')
sector_67Y_list = [sector_y_esmm, sector_6_esmm]
sector_67Y = gpd.GeoDataFrame(
    pd.concat(sector_67Y_list, ignore_index=True),
    crs='epsg:4326'
).dissolve()
sectors_list = [
    {
        "name": "sector_67Y",
        "sector_gpd": sector_67Y
    },
    {
        "name": "sector_w_esmm",
        "sector_gpd": sector_w_esmm
    }
]

reference_sector = sector_y_esmm
m = folium.Map(
    [
        reference_sector.geometry[0].centroid.y,
        reference_sector.geometry[0].centroid.x, 
    ], 
    zoom_start=6, tiles='cartodbpositron')
folium.GeoJson(
    sector_67Y,
    style_function=lambda features:{'color': 'blue'}
).add_to(m)
folium.GeoJson(
    sector_w_esmm,
    style_function=lambda features:{'color': 'black'}
).add_to(m)
folium.LatLngPopup().add_to(m)
m

# # Specify the directory of raw files

data_folders = [
    './raw_data/scat20161015_20161021/',
    './raw_data/scat20161112_20161118/',
    './raw_data/scat20161210_20161216/',
    './raw_data/scat20170107_20170113/',
    './raw_data/scat20170215_20170221/',
    './raw_data/scat20170304_20170310/',
    './raw_data/scat20170401_20170407/',
    './raw_data/scat20170429_20170505/',
    './raw_data/scat20170527_20170602/',
    './raw_data/scat20170624_20170630/',
    './raw_data/scat20170722_20170728/',
    './raw_data/scat20170819_20170825/',
    './raw_data/scat20170916_20170922/'
]



# # Filter relevant flights

# +
# %%time

processed_data_save_in = "./processed_data"
if not os.path.exists(processed_data_save_in):
    os.mkdir(processed_data_save_in)

excluded_files = [
    'airspace.json',
    'grib_meteo.json'
]

for each_week_folder in data_folders:
    print('\n#####\n')
    files_to_generate = []
    this_week = each_week_folder.split('/')[-2]
    for each_sector in sectors_list:
        this_sector_folder = f"{processed_data_save_in}/{each_sector['name']}"
        if not os.path.exists(this_sector_folder):
            print(f"The directory '{this_sector_folder}' does not exist, creating the folder ...")
            os.mkdir(this_sector_folder)
        
        this_file_name = f"{this_sector_folder}/intermediate_data/{this_week}_filtered.geojson"
        if os.path.isfile(this_file_name):
            print(f"File '{this_file_name}' already exists, skipping to next iteration")
        else:
            print(f"File '{this_file_name}' does not exists, adding into task list")
            files_to_generate.append([this_file_name, each_sector])
    
    if len(files_to_generate) == 0:
        print(f"All files for week '{this_week}' has been generated, moving on to the next week")
        continue
        
    print(f"Processing week '{this_week}'")
    flight_point_rows = []
    for each_raw_file in os.listdir(each_week_folder):
        if each_raw_file in excluded_files:
            continue
        with open(each_week_folder + each_raw_file, 'r') as f:
            this_flight = json.load(f)
        for each_point in this_flight['plots']:
            flight_point_rows.append({
                'id': this_flight['id'],
                'lat': each_point['I062/105']['lat'],
                'lon': each_point['I062/105']['lon']
            })
    this_df_points = pd.DataFrame(flight_point_rows)
    this_gdf_points = gpd.GeoDataFrame(
        this_df_points,
        geometry=gpd.points_from_xy(
            x=this_df_points.lon,
            y=this_df_points.lat
        ),
        crs='EPSG:4326'
    )
    this_gdf_lines = gpd.GeoDataFrame(
        this_gdf_points.groupby(['id'])['geometry'].apply(lambda x: LineString(x.tolist())),
        geometry='geometry',
        crs="EPSG:4326"
    ).reset_index()
    
    for each_output_task in files_to_generate:
        this_sector_gpd = each_output_task[1]['sector_gpd']
        this_gdf_lines['in_sector'] = this_gdf_lines.geometry\
            .apply(lambda x: this_sector_gpd.geometry[0].intersects(x))
        this_gdf_lines.loc[this_gdf_lines.in_sector==True].to_file(
            each_output_task[0],
            driver='GeoJSON'
        )
        print(f"Generated and saved geojson '{each_output_task[0]}'")
# -





# # Identify Feasible Flights

# +
d_format = "%Y-%m-%dT%H:%M:%S.%f"
d_format_short = "%Y-%m-%dT%H:%M:%S"

def get_correct_update(fpl_plan_updates, entry_time_str , minutes_ahead=30):
    latest = datetime.datetime.strptime("1900-01-01T00:00:00.00000", d_format)
    res = False
    entry_time = datetime.datetime.strptime(entry_time_str, d_format)
    threshold_time = entry_time - datetime.timedelta(0,minutes_ahead*60)
    for update in fpl_plan_updates:
        time_str = update["time_stamp"]
        if "." not in time_str:
            time_str += ".00"
        timestamp = datetime.datetime.strptime(time_str, d_format)
        if timestamp <= threshold_time:
            if latest < timestamp:
                res = update
                latest = timestamp

    return res


def get_correct_predicted_traj(predicted_traj, entry_time_str , minutes_ahead=30):
    latest = datetime.datetime.strptime("1900-01-01T00:00:00.00000", d_format)
    # issue: what if the oldest plan is not old enough ?? 
    res = False
    pt_timestamp = None
    entry_time = datetime.datetime.strptime(entry_time_str, d_format)
    threshold_time = entry_time - datetime.timedelta(0,minutes_ahead*60)
    for pt_elem in predicted_traj:
        time_str = pt_elem["time_stamp"]
        if "." not in time_str:
            time_str += ".00"
        timestamp = datetime.datetime.strptime(time_str, d_format)
        if timestamp <= threshold_time:
            if latest < timestamp:
                res = pt_elem
                latest = timestamp
                pt_timestamp = time_str
    return res, pt_timestamp
        
        
def get_fp_route(flight_plan_update):
    route_str = flight_plan_update['icao_route']
    plan_route_strings = flight_plan_update["icao_route"].split(" ")
    route_list = []
    for elem in plan_route_strings:
        if elem in air_space_dict:
            route_list.append(Point(air_space_dict[elem]['lon'], air_space_dict[elem]['lat']))
        
    if len(route_list) > 1:
        return LineString(route_list)
    else:
        return None
    
def get_route_from_traj(pred_traj):
    route_list = []
    for point in pred_traj['route']:
        route_list.append(Point(point['lon'], point['lat']))
    return LineString(route_list)

    
def get_entry(line, sector):
    intersection_line = line.intersection(sector)
    if isinstance(intersection_line, MultiLineString):
        intersection_line = intersection_line.geoms[0]
    intersections = intersection_line.xy
    if len(intersections[0]) == 0:
        return None, None
    entry_lon = intersections[0][0]
    entry_lat = intersections[1][0]
    return entry_lon, entry_lat

def get_position_at_prediction_time(prediction_time: datetime.datetime, input_flight):
    prev_point = {}
    prev_timestamp = None
    for each_point in input_flight['plots']:
        try:
            point_timestamp = datetime.datetime.strptime(
                each_point['time_of_track'], '%Y-%m-%dT%H:%M:%S.%f')
        except:
            point_timestamp = datetime.datetime.strptime(
                each_point['time_of_track'], '%Y-%m-%dT%H:%M:%S')
        if point_timestamp > prediction_time:
            return prev_point['I062/105']['lon'], prev_point['I062/105']['lat']
        prev_point = each_point
        prev_timestamp = point_timestamp
    return None, None

def get_entry_time(A,C,A_time_str,C_time_str,B):
    time_A = datetime.datetime.strptime(A_time_str,d_format_short) 
    time_C = datetime.datetime.strptime(C_time_str,d_format_short)
    time_B = time_A + datetime.timedelta(seconds=(time_C-time_A).total_seconds() * (A.distance(B) / A.distance(C)))
    return time_B.strftime(d_format)
# -



# +
# %%time
buffer_list = [15]

for each_week_folder in data_folders:
    files_to_generate = []
    this_week = each_week_folder.split('/')[-2]
    print('\n#####\n')
    print(f"Inspecting week '{this_week}'")
    
    for each_sector in sectors_list:
        this_sector_folder = f"{processed_data_save_in}/{each_sector['name']}"     
        this_filtered_flights_file = f"{this_sector_folder}/intermediate_data/{this_week}_filtered.geojson"
        this_filtered_flights = None
        sector_geom = each_sector['sector_gpd'].geometry[0]
        
        for each_buffer in buffer_list:
            entries = []
            entries_id = []
            unavailable_pt = []
            plot_start_in_sector = []
            not_planned_to_enter = []
            file_scanned = 0
            
            this_filter_stats_file = f"{this_sector_folder}/intermediate_data/{this_week}_buffer{each_buffer}_stats.json"
            this_filter_result_file = f"{this_sector_folder}/intermediate_data/{this_week}_buffer{each_buffer}_results.csv"
            if os.path.isfile(this_filter_result_file):
                print(f"File '{this_filter_result_file}' already exists, skipping to next iteration")
                continue
            if this_filtered_flights is None:
                with open(this_filtered_flights_file, 'r') as f:
                    this_filtered_flights = gpd.read_file(f)
                    
            for each_raw_file in os.listdir(each_week_folder):
                if each_raw_file.strip('.json') not in map(str, this_filtered_flights.id.tolist()):
                    continue
                file_scanned += 1
                with open(each_week_folder + each_raw_file, 'r') as f:
                    this_flight = json.load(f)
                plots = this_flight['plots']
                for i, p in enumerate(plots):
                    point = Point(p['I062/105']['lon'], p['I062/105']['lat'])
                    actual_entry_time = None
                    if sector_geom.contains(point):
                        # if the plot already starts in the sector, the flight is ignored
                        if i == 0:
                            plot_start_in_sector.append(each_raw_file.strip('.json'))
                            break
                        prelim_pred_traj = None # initialize to None so no value is carrid forward
                        pred_traj = None # initialize to None so no value is carrid forward
                        last_outside_point = Point(plots[i-1]['I062/105']['lon'], plots[i-1]['I062/105']['lat'])
                        line = LineString([last_outside_point, point])
                        actual_entry_time = p['time_of_track']
                        assert actual_entry_time is not None
                        if "." not in actual_entry_time:
                            actual_entry_time += ".00"
                        entry_lon, entry_lat = get_entry(line, sector_geom)
                        # this needs to be changed if you want to use flightplan updates
                        prelim_pred_traj, prelim_pt_ts = get_correct_predicted_traj(
                            this_flight['predicted_trajectory'], 
                            actual_entry_time, 
                            each_buffer
                        )
                        
                        if not prelim_pred_traj:
                            #print(f"{file} has no predicted trajectory that is old enough")
                            unavailable_pt.append(each_raw_file.strip('.json'))
                            break
                        
                        prelim_routes = prelim_pred_traj['route']
                        forecasted_entry_time = None
                        prev_pt_point = None
                        pt_point = None
                        prev_eto = None
                        this_entry_lon = None
                        this_entry_lat = None
                        for pt_i, pt_p in enumerate(prelim_routes):
                            pt_point = Point(pt_p['lon'], pt_p['lat'])
                            if prev_pt_point == None:
                                prev_pt_point = pt_point
                                prev_eto = pt_p['eto']
                                continue
                            this_pt_line = LineString([prev_pt_point, pt_point])
                            if this_pt_line.intersects(sector_geom):
                                this_entry_lon, this_entry_lat = get_entry(
                                        this_pt_line, sector_geom)
                                assert prev_eto is not None
                                forecasted_entry_time = get_entry_time(
                                        prev_pt_point,
                                        pt_point,
                                        prev_eto,
                                        pt_p['eto'],
                                        Point(this_entry_lon, this_entry_lat)
                                    )
                                break
                            prev_pt_point = pt_point
                            prev_eto = pt_p['eto']
                        #try:
                        #    assert forecasted_entry_time is not None
                        #except:
                        #    print(this_flight['id'])
                        #    print(prelim_pt_ts)
                        #    print(prev_pt_point, pt_point)
                        #finally:
                        #    assert forecasted_entry_time is not None
                        #assert forecasted_entry_time is not None
                        if forecasted_entry_time is not None:
                            if "." not in forecasted_entry_time:
                                forecasted_entry_time += ".00"
                            pred_traj, pt_ts = get_correct_predicted_traj(
                                this_flight['predicted_trajectory'],
                                forecasted_entry_time,
                                each_buffer
                            )

                            if not pred_traj:
                                #print(f"{file} has no predicted trajectory that is old enough")
                                unavailable_pt.append(each_raw_file.strip('.json'))
                                break
                            this_prediction_time = datetime.datetime.strptime(
                                                        forecasted_entry_time, d_format
                                                    ) - datetime.timedelta(minutes=each_buffer)
                            this_prediction_time_str = this_prediction_time.strftime(d_format)
                        else:
                            pred_traj = prelim_pred_traj
                            pt_ts = prelim_pt_ts
                            this_prediction_time = datetime.datetime.strptime(
                                                        actual_entry_time, d_format
                                                    ) - datetime.timedelta(minutes=each_buffer)
                            this_prediction_time_str = this_prediction_time.strftime(d_format)
                            
                        fp_entry_lon, fp_entry_lat = get_entry(get_route_from_traj(pred_traj), sector_geom)
                        entries_id.append(this_flight['id'])
                        actual_lon_at_pred, actual_lat_at_pred = get_position_at_prediction_time(
                            this_prediction_time, this_flight)
                        this_entry_deviation = None if fp_entry_lon is None else geo_distance.geodesic(
                                                    (entry_lat, entry_lon),
                                                    (fp_entry_lat, fp_entry_lon)).km
                        entries.append({
                            'id': this_flight['id'],
                            'predicted_trajectory_ts': pt_ts,
                            'actual_entry_time': actual_entry_time,
                            'forecasted_entry_time': forecasted_entry_time,
                            'prediction_time': this_prediction_time_str,
                            'plt_entry_lon':entry_lon, 
                            'plt_entry_lat':entry_lat,
                            'fp_entry_lon': fp_entry_lon,
                            'fp_entry_lat': fp_entry_lat,
                            'at_pred_lon': actual_lon_at_pred,
                            'at_pred_lat': actual_lat_at_pred,
                            'entry_deviation': this_entry_deviation
                        })
                        break
            # here is each buffer level
            this_stats = {
                'time_buffer': each_buffer,
                'file_scanned': file_scanned,
                'unavalable_pt': unavailable_pt,
                'plot_start_in_sector': plot_start_in_sector,
                'not_planned_to_enter': not_planned_to_enter,
                'entries_id': entries_id
            }
            with open(this_filter_stats_file, 'w') as f_stats:
                json.dump(this_stats, f_stats, indent=4)
            print(f"Filter stats file '{this_filter_stats_file}' has been generated and saved")
            df_feasible_flight = pd.DataFrame(entries)
            df_feasible_flight.to_csv(this_filter_result_file, index=False)
            print(f"Filter results file '{this_filter_result_file}' has been generated and saved")
# -



# # Combining output files

# +
# %%time
buffer_list = [15]

for each_sector in sectors_list:
    this_sector_folder = f"{processed_data_save_in}/{each_sector['name']}"   
    for each_buffer in buffer_list:
        this_combined_file_name = f"{this_sector_folder}/sector_{each_sector['name']}_buffer{each_buffer}_combined_results.csv"
        if os.path.isfile(this_combined_file_name):
            print(f"Combined file '{this_combined_file_name}' already exist, skipping to the next one ...")
            continue
        this_df_list = []
        for each_week_folder in data_folders:
            this_week = each_week_folder.split('/')[-2]
            this_sector_folder_input = f"{processed_data_save_in}/{each_sector['name']}/intermediate_data"
            this_filter_result_file = f"{this_sector_folder_input}/{this_week}_buffer{each_buffer}_results.csv"
            assert os.path.isfile(this_filter_result_file)
            this_df_list.append(pd.read_csv(this_filter_result_file))
        this_combined_df = pd.concat(this_df_list, ignore_index=True)
        this_combined_df.to_csv(this_combined_file_name, index=False)
        print(f"Combined file '{this_combined_file_name}' has been generated and saved")
# -




