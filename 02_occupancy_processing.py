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

import json
import os
import geopandas as gpd
from shapely.geometry import Point, LineString,box,MultiLineString
import datetime
import time
from shapely import wkt


raw_data_path = "D:/lfv-main-data/data/"
sector_name  = "SectorW"
processed_data_path = ""




# # Create Trajectory table
# create a Geo data frame, with route/trajectory and entry_time into the grid_space relating to a sector. Route and time relates to the oldest flight plan entry of each flight. If trajectory.csv is already present, for the respective folder it will be loaded, created otherwise (this can take a while)*

grid_frame_dic =  {"SectorW": [13.5,56.5,19,60]}
#TODO enter other sectors

a,b,c,d = grid_frame_dic[sector_name]
grid_frame =  box(a,b,c,d)

# +
d_format = "%Y-%m-%dT%H:%M:%S.%f"
d_format_short = "%Y-%m-%dT%H:%M:%S"

class StartsInSectorException(Exception):
    pass

class NoExitFromSectorException(Exception):
    pass

class NoIntersectionException(Exception):
    pass

def get_entry(line, sector):
    intersection_line = line.intersection(sector)
    if isinstance(intersection_line, MultiLineString):
        intersection_line = intersection_line.geoms[0]
    intersections = intersection_line.xy
    if len(intersections[0]) == 0:
        raise NoIntersectionException
    entry_lon = intersections[0][0]
    entry_lat = intersections[1][0]
    return entry_lon, entry_lat

def get_entry_time(A,C,A_time_str,C_time_str,B):
    time_A = datetime.datetime.strptime(A_time_str,d_format_short) 
    time_C = datetime.datetime.strptime(C_time_str,d_format_short)
    time_B = time_A + datetime.timedelta(seconds=(time_C-time_A).total_seconds() * (A.distance(B) / A.distance(C)))
    return time_B.strftime(d_format)

def find_first_and_time(route,sector):
    points, times = [],[]
    for elem in route:
        points.append(Point(elem['lon'],elem['lat']))
        times.append(elem['eto'])
    if sector.contains(points[0]):
        return route[0]['lon'], route[0]['lat'],route[0]['eto']
    for j,p in enumerate(points[1:]):
        i = j-1
        line = LineString([points[i-1],p])
        if line.intersects(sector):
            entry_lon, entry_lat = get_entry(line, sector)
            entry_time = get_entry_time(points[i-1],p,times[i-1],times[i],Point(entry_lon, entry_lat))
            return entry_lon, entry_lat, entry_time
    raise NoIntersectionException


# -

folders = []
for d in os.listdir(raw_data_path):
    folders.append(raw_data_path+d+"/")

if "trajectories.csv" not in os.listdir(f"{processed_data_path}/{sector_name}/intermediate/")_ 
    predicted_trajectories = []
    start =  time.time()
    for each_folder in folders:
        print(each_folder)
        for each_file in os.listdir(each_folder):
            if each_file in ['airspace.json',"grib_meteo.json"]:
                continue
            with open(each_folder + each_file, 'r') as f:
                flight = json.load(f)
            if len(flight['predicted_trajectory']) == 0:
                continue
            ind,pt = 0, flight['predicted_trajectory'][0]
            route_lst = []
            for p in pt['route']:
                route_lst.append(Point(p['lon'],p['lat']))
            try:
                entry_lon, entry_lat, entry_time = find_first_and_time(pt['route'],grid_frame)
            except NoIntersectionException:
                continue
            predicted_trajectories.append({'flightID':flight['id'], 'time_stamp':pt['time_stamp'], "entry_time":entry_time ,"index":ind, 
                                           "geometry":LineString(route_lst)})
        print(time.time() - start)
    crs = 'epsg:4326'
    flight_plan_gdf = gpd.GeoDataFrame(predicted_trajectories,crs = crs, geometry="geometry")
    flight_plan_gdf.to_csv(f"{processed_data_path}/{sector_name}/intermediate/trajectories.csv")


pred_traj_pd = pd.read_csv(f"{processed_data_path}/{sector_name}/intermediate/trajectories.csv",index_col=0)
pred_traj_pd['geometry'] = pred_traj_pd['geometry'].apply(wkt.loads)


# # Trajectory
# creates matrix for every flight, which represents if a certain grid within the gridspace is ever crossed by the trajectory of a flight (refering to the trajectory.csv)

def create_grid(lon_min, lon_max, lonN, lat_min, lat_max, latN):
    lon_step_size = (lon_max - lon_min) / lonN
    lat_step_size = (lat_max - lat_min) / latN
    tiles = []
    lat = lat_max
    for i in range(latN):
        lon = lon_min
        lat_next = lat-lat_step_size
        for j in range(lonN):
            lon_next = lon+lon_step_size
            tiles.append(box(lon, lat, lon_next, lat_next))
            lon = lon_next 
        lat = lat_next
    grid_frame = box(lon_min,lat_min,lon_max,lat_max)
    return tiles , grid_frame 


a,b,c,d = grid_frame_dic[sector_name]
grid_cells,grid_frame = create_grid(a,c,10,b,d,10)
crs = 'epsg:4326'
grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], 
                                 crs=crs)


# %%time
filtered_path = f'./{processed_data_path}/{sector_name}/final??TODO.csv'
df_filtered = pd.read_csv(filtered_path)
df_filtered['forecasted_entry_time'] = pd.to_datetime(df_filtered['forecasted_entry_time'])
df_filtered.head()

# +
relevant_ids = df_filtered['id'].values

pt_gdf = pred_traj_pd.loc[pred_traj_pd['flightID'].isin(relevant_ids)].copy()

pt_gdf['time_stamp'] = pd.to_datetime(pt_gdf['time_stamp'])
# -

# %%time
if "trajectory.csv" not in os.listdir(f"{processed_data_path}/{sector_name}/trajectory/"):
    print("start creating trajectory matrices")
    pt_occupancies = []
    for id_ , flight in df_filtered.iterrows():
        this_matrix = np.zeros(len(grid))
        if flight['id'] not in pt_gdf['flightID'].values:
            pt_occupancies.append(this_matrix)
            print(flight['id'],"has no grid intersection")
            continue
        route = pt_gdf.loc[pt_gdf['flightID'] == flight['id']]['geometry'].values[0]
        intersections = grid.intersects(route)
        this_matrix += intersections
        pt_occupancies.append(this_matrix)
    colnames = ["flight_id"] + list(range(len(grid)))
    pt_occ_df = pd.DataFrame(np.insert(pt_occupancies,0,df_filtered['id'].values,axis=1), columns =colnames)
    pt_occ_df.to_csv(f"{processed_data_path}/{sector_name}/trajectory/trajectory.csv",index=False)
    print("trajectory.csv created")
    pt_occ_df
else:
    print("trajectory.csv already existend for sector")


# # Occupancy
# Create the occupancy matrixes at different timepoints (see deltas_from_et). Variable refers to minutes after the expected sector entry of a flight.
#
# To recreate the matrices ensure that there is an empty folder occupancy/ in the respective processed_data/sector directory. 

# +
def point_at_time(trajectory, timestamp):
    last_point = None
    for point in trajectory:
        eto = datetime.strptime(point['eto'],"%Y-%m-%dT%H:%M:%S")
        if eto > timestamp and last_point:
            last_eto =  datetime.strptime(last_point['eto'],"%Y-%m-%dT%H:%M:%S")
            if eto == last_eto:
                continue
            factor = (timestamp -last_eto).seconds / (eto - last_eto).seconds
            res_lon = last_point['lon'] + factor*(point['lon'] - last_point['lon'])
            res_lat = last_point['lat'] + factor*(point['lat'] - last_point['lat'])
            return Point(res_lon, res_lat)
        else:
            last_point = point
    return None

def find_folder(path, file):
    for folder in os.listdir(path):
        if file in os.listdir(path+folder):
            return path+folder+"/"
    return ""  


# +
video_occ_path = f"{processed_data_path}/{sector_name}/occupancy/"
deltas_from_et = [0,5,10]

# %%time
if len(os.listdir(video_occ_path)) == 0:
    print("start creating occupancy matrices")
    os.mkdir(f"{video_occ_path}occs")
    os.mkdir(f"{video_occ_path}ids")
    c = 0
    point_delta_dict = {}
    for id_ , flight in df_filtered.iterrows():
        c+=1
        folder_path = find_folder(raw_data_path, str(flight['id']) + ".json")
        entry_time = flight['forecasted_entry_time']
        point_delta_dict[flight['id']] = {}
        for fid_pt in relevant_pts_ff[flight['id']]:
            point_delta_dict[flight['id']][fid_pt] = {}
            with open(folder_path+ str(fid_pt) +".json", 'r') as f:
                flight_json = json.load(f)
            pt_route = flight_json['predicted_trajectory'][0]['route']
            for delta in deltas_from_et:
                point = point_at_time(pt_route, entry_time + timedelta(minutes=delta))
                point_delta_dict[flight['id']][fid_pt][delta] = point
        if (c%1000)==0:
            print(c/len(df_filtered) * 100, "%")
            all_occs = []
            for fid, point_dict in point_delta_dict.items():
                occs = []
                for delta in deltas_from_et:
                    this_delta_occ = np.zeros(100)
                    for pt_ids,delta_dict in point_dict.items():
                        if delta_dict[delta]:
                            this_delta_occ += grid.intersects(delta_dict[delta])
                    occs.append(this_delta_occ)
                all_occs.append(occs)
            all_occs = np.array(all_occs)
            np.save(video_occ_path+"occs/"+str(c/1000),all_occs)
            point_delta_dict = {}
    all_occs = []
    for fid, point_dict in point_delta_dict.items():
        occs = []
        for delta in deltas_from_et:
            this_delta_occ = np.zeros(100)
            for pt_ids,delta_dict in point_dict.items():
                if delta_dict[delta]:
                    this_delta_occ += grid.intersects(delta_dict[delta])
            occs.append(this_delta_occ)
        all_occs.append(occs)
    all_occs = np.array(all_occs)
    np.save(video_occ_path+"occs/"+f"{int(len(df_filtered)//1000)+1}.0",all_occs)
    np.save(video_occ_path+"ids", df_filtered['id'].values)
    print("finished creating occupancy matrices")
else:
    print("occupancy matrices already exist for the sector. To recreate, delete occs and ids folder")
