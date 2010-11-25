#!/usr/bin/python
import nio
import numpy as N
import os
def sonic(filename):
    '''
        Create .nc files from RMY Sonic Anemometer ASCII Data from the Lake Thunderbird Micronet
    '''
#
#   Open ASCII File
#
    try:
        date = N.loadtxt(filename, usecols=[0], dtype=str, comments='%', delimiter=',')
        u1,u2,u3,u4,u5,v1,v2,v3,v4,v5,w1,w2,w3,w4,w5,T1,T2,T3,T4,T5,voltage= \
            N.loadtxt(filename, usecols=N.arange(1,22), comments='%', delimiter=',',unpack=True)
        u=N.r_[u1,u2,u3,u4,u5]
        v=N.r_[v1,v2,v3,v4,v5]
        w=N.r_[w1,w2,w3,w4,w5]
        T=N.r_[T1,T2,T3,T4,T5]

        dt = N.dtype([('u',N.float),('v',N.float),('w',N.float),
                      ('T',N.float)])       
        data=N.array(zip(u,v,w,T), dtype=dt).reshape(5,-1)

        dt = N.dtype([('date',object),('voltage',N.float)])
        ext = N.array(zip(date,voltage), dtype=dt) 


        netcdf_name = os.path.splitext(os.path.basename(filename))[0]+'.nc'
        netcdf_loc = '/micronet/python/data/ltmsonic/netcdf'
        netcdf_fpath = os.path.join(netcdf_loc,netcdf_name)
#
#       Open netCDF file and let the fun begin!
#
        f = nio.open_file(netcdf_fpath, mode='c')
#
#       Set dimensions
#
        f.create_dimension('station',5)
        f.create_dimension('record_length',18000)
#
#       Create Variables
#
#        sdi_var=f.create_variable('station_id','i',('station',))
#        lat_var=f.create_variable('latitude','f',('station',))
#        long_var=f.create_variable('longitude','f',('station',))
#        alt_var=f.create_variable('altitude','f',('station',))
#        time_var=f.create_variable('time','i',('station','record_length',))
        u_var=f.create_variable('u','f',('station','record_length'))
        v_var=f.create_variable('v','f',('station','record_length'))
        w_var=f.create_variable('w','f',('station','record_length'))
        T_var=f.create_variable('T','f',('station','record_length'))
#        bat_v_var=f.create_variable('bat_v','f',('record_length',))
#
#       Assign attributes to variables
#
#        f.variables['station_id'].long_name = 'tower level'

#        f.variables['latitude'].units = 'degrees_north'
#        f.variables['latitude'].long_name = 'latitude degrees_north'

#        f.variables['longitude'].units = 'degrees_east'
#        f.variables['longitude'].long_name = 'longitude degrees_east'

#        f.variables['altitude'].units = 'meters'
#        f.variables['altitude'].positive = 'up'
#        f.variables['altitude'].long_name = 'height above ground level'

#        f.variables['time'].units = 'seconds since 1970-01-01 00:00:00.000 UTC'
#        f.variables['time'].long_nameunits = 'seconds since 1970-01-01 00:00:00.000 UTC'

        f.variables['u'].units = 'm/s'
        f.variables['u'].long_name = 'East-west component of velocity'
#        f.variables['u'].missing_value = '-999'

        f.variables['v'].units = 'm/s'
        f.variables['v'].long_name = 'North-south component of velocity'
#        f.variables['v'].missing_value = '-999'

        f.variables['w'].units = 'm/s'
        f.variables['w'].long_name = 'vertical component of velocity (parallel to gravity)'
#        f.variables['w'].missing_value = '-999'

        f.variables['T'].units = 'Kelvin'
        f.variables['T'].long_name = 'Temperature in Kelvin'
#        f.variables['T'].missing_value = '-999'
#
#       Assign values to variables
#
#        pnt=post_num
#        latt=lat_array
#        lont=lon_array
#        altt=N.array([1.5,3.0,6.0,10.0,15.0])
#        outtimet=outtime_array

#        for i in range(0,287):
#            post_num=N.c_[post_num,pnt]
#            lat_array=N.c_[lat_array,latt]
#            lon_array=N.c_[lon_array,lont]
#            alt_array=N.c_[alt_array,altt]

#        outtime_array=ext['date'][0]
#        for i in range(1,5):
#            outtime_array=N.c_[outtime_array,ext['date'][i]]
#        sdi_var.assign_value(N.array([1,2,3,4,5]))
#        lat_var.assign_value(lat_array)
#        long_var.assign_value(lon_array)
#        alt_var.assign_value(alt_array)
#        time_var.assign_value(outtime_array)
        u_var.assign_value(data['u'].astype('float32'))
        v_var.assign_value(data['v'].astype('float32'))
        w_var.assign_value(data['w'].astype('float32'))
        T_var.assign_value(data['T'].astype('float32'))
#        bat_v_var.assign_value(ext['voltage'])
#
#       Set global attributes
#
        setattr(f,'title','Lake Thunderbird Micronet sonic anemometer time series')
#        setattr(f,'Conventions','Unidata Observation Dataset v1.0')
#        setattr(f,'description','Lake Thunderbird Micronet Temperature/RH Daily Time Series')
#        setattr(f,'time_coordinate','time')
#        setattr(f,'cdm_datatype','Station')
#        setattr(f,'stationDimension','station')
#        setattr(f,'station_id_variable','station_id')
#        setattr(f,'latitude_coortinate','latitude')
#        setattr(f,'longitude_coortinate','longitude')
#        setattr(f,'altitude_coortinate','altitude')
#        setattr(f,'geospatial_lat_max',N.array2string(lat_array.max()))
#        setattr(f,'geospatial_lat_min',N.array2string(lat_array.min()))
#        setattr(f,'geospatial_lon_max',N.array2string(lon_array.max()))
#        setattr(f,'geospatial_lon_min',N.array2string(lon_array.min()))
#        setattr(f,'time_coverage_start',str(int(outtime[0]))+" seconds since 1970-01-01 00:00:00.000 UTC")
#        setattr(f,'time_coverage_end',str(int(outtime[-1]))+" seconds since 1970-01-01 00:00:00.000 UTC")
#        setattr(f,'observationDimension','record_length')
#
#       Close netCDF file
#
        f.close()
    except IOError:
        print '%s does not exist\n'%filename
        raise
def sonic_day(data_dir):
    '''
        Create day long .nc files from RMY Sonic Anemometer ASCII Data from the Lake Thunderbird Micronet
    '''
    files = os.listdir(data_dir)
    num_files = len(files)
    files.sort()
#
#   Open ASCII File
#
    hour_count = 0
    for i in range(0,num_files):
        try:
            ascii_files = os.path.join(data_dir,files[i])
            netcdf_filename = files[i][0:10]+'.nc'
            if hour_count != 0:
                print "    "+files[i]
                date = N.loadtxt(ascii_files, usecols=[0], dtype=str, comments='%', delimiter=',')
                u1,u2,u3,u4,u5,v1,v2,v3,v4,v5,w1,w2,w3,w4,w5,T1,T2,T3,T4,T5,voltage= \
                    N.loadtxt(ascii_files, usecols=N.arange(1,22), comments='%', delimiter=',',unpack=True)

                u1a=N.r_[u1a,u1]
                u2a=N.r_[u2a,u2]
                u3a=N.r_[u3a,u3]
                u4a=N.r_[u4a,u4]
                u5a=N.r_[u5a,u5]
                v1a=N.r_[v1a,v1]
                v2a=N.r_[v2a,v2]
                v3a=N.r_[v3a,v3]
                v4a=N.r_[v4a,v4]
                v5a=N.r_[v5a,v5]
                w1a=N.r_[w1a,w1]
                w2a=N.r_[w2a,w2]
                w3a=N.r_[w3a,w3]
                w4a=N.r_[w4a,w4]
                w5a=N.r_[w5a,w5]
                T1a=N.r_[T1a,T1]
                T2a=N.r_[T2a,T2]
                T3a=N.r_[T3a,T3]
                T4a=N.r_[T4a,T4]
                T5a=N.r_[T5a,T5]
                hour_count=hour_count+1
            elif hour_count == 0:
                print files[i]
                date = N.loadtxt(ascii_files, usecols=[0], dtype=str, comments='%', delimiter=',')
                u1a,u2a,u3a,u4a,u5a,v1a,v2a,v3a,v4a,v5a,w1a,w2a,w3a,w4a,w5a,T1a,T2a,T3a,T4a,T5a,voltagea= \
                    N.loadtxt(ascii_files, usecols=N.arange(1,22), comments='%', delimiter=',',unpack=True)
                hour_count=hour_count+1

        except IOError:
            print '%s does not exist\n'%filename
            raise
        if hour_count == 24:

            hour_count = 0

            u=N.r_[u1a,u2a,u3a,u4a,u5a]
            v=N.r_[v1a,v2a,v3a,v4a,v5a]
            w=N.r_[w1a,w2a,w3a,w4a,w5a]
            T=N.r_[T1a,T2a,T3a,T4a,T5a]


            dt = N.dtype([('u',N.float),('v',N.float),('w',N.float),
                          ('T',N.float)])       
            data=N.array(zip(u,v,w,T), dtype=dt).reshape(5,-1)

            dt = N.dtype([('date',object),('voltage',N.float)])
            ext = N.array(zip(date,voltage), dtype=dt) 

            netcdf_loc = '/micronet/python/data/ltmsonic/netcdf_day'
            netcdf_fpath = os.path.join(netcdf_loc,netcdf_filename)
#
#           Open netCDF file and let the fun begin!
#
            f = nio.open_file(netcdf_fpath, mode='c')
#
#          Set dimensions
#
            f.create_dimension('station',5)
            f.create_dimension('record_length',432000)
#
#           Create Variables
#
#            sdi_var=f.create_variable('station_id','i',('station',))
#            lat_var=f.create_variable('latitude','f',('station',))
#            long_var=f.create_variable('longitude','f',('station',))
#            alt_var=f.create_variable('altitude','f',('station',))
#            time_var=f.create_variable('time','i',('station','record_length',))
            u_var=f.create_variable('u','f',('station','record_length'))
            v_var=f.create_variable('v','f',('station','record_length'))
            w_var=f.create_variable('w','f',('station','record_length'))
            T_var=f.create_variable('T','f',('station','record_length'))
#            bat_v_var=f.create_variable('bat_v','f',('record_length',))
#
#           Assign attributes to variables
#
#            f.variables['station_id'].long_name = 'tower level'

#            f.variables['latitude'].units = 'degrees_north'
#            f.variables['latitude'].long_name = 'latitude degrees_north'

#            f.variables['longitude'].units = 'degrees_east'
#            f.variables['longitude'].long_name = 'longitude degrees_east'

#            f.variables['altitude'].units = 'meters'
#            f.variables['altitude'].positive = 'up'
#            f.variables['altitude'].long_name = 'height above ground level'

#            f.variables['time'].units = 'seconds since 1970-01-01 00:00:00.000 UTC'
#            f.variables['time'].long_nameunits = 'seconds since 1970-01-01 00:00:00.000 UTC'

            f.variables['u'].units = 'm/s'
            f.variables['u'].long_name = 'East-west component of velocity'
#            f.variables['u'].missing_value = '-999'

            f.variables['v'].units = 'm/s'
            f.variables['v'].long_name = 'North-south component of velocity'
#            f.variables['v'].missing_value = '-999'

            f.variables['w'].units = 'm/s'
            f.variables['w'].long_name = 'vertical component of velocity (parallel to gravity)'
#            f.variables['w'].missing_value = '-999'

            f.variables['T'].units = 'Kelvin'
            f.variables['T'].long_name = 'Temperature in Kelvin'
#            f.variables['T'].missing_value = '-999'
#
#           Assign values to variables
#
#            pnt=post_num
#            latt=lat_array
#            lont=lon_array
#            altt=N.array([1.5,3.0,6.0,10.0,15.0])
#            outtimet=outtime_array

#           for i in range(0,287):
#                post_num=N.c_[post_num,pnt]
#                lat_array=N.c_[lat_array,latt]
#                lon_array=N.c_[lon_array,lont]
#                alt_array=N.c_[alt_array,altt]

#            outtime_array=ext['date'][0]
#            for i in range(1,5):
#                outtime_array=N.c_[outtime_array,ext['date'][i]]
#            sdi_var.assign_value(N.array([1,2,3,4,5]))
#            lat_var.assign_value(lat_array)
#            long_var.assign_value(lon_array)
#            alt_var.assign_value(alt_array)
#            time_var.assign_value(outtime_array)
            u_var.assign_value(data['u'].astype('float32'))
            v_var.assign_value(data['v'].astype('float32'))
            w_var.assign_value(data['w'].astype('float32'))
            T_var.assign_value(data['T'].astype('float32'))
#            bat_v_var.assign_value(ext['voltage'])
#
#           Set global attributes
#
            setattr(f,'title','Lake Thunderbird Micronet sonic anemometer time series')
#           setattr(f,'Conventions','Unidata Observation Dataset v1.0')
#           setattr(f,'description','Lake Thunderbird Micronet Temperature/RH Daily Time Series')
#           setattr(f,'time_coordinate','time')
#           setattr(f,'cdm_datatype','Station')
#           setattr(f,'stationDimension','station')
#           setattr(f,'station_id_variable','station_id')
#           setattr(f,'latitude_coortinate','latitude')
#           setattr(f,'longitude_coortinate','longitude')
#           setattr(f,'altitude_coortinate','altitude')
#           setattr(f,'geospatial_lat_max',N.array2string(lat_array.max()))
#           setattr(f,'geospatial_lat_min',N.array2string(lat_array.min()))
#           setattr(f,'geospatial_lon_max',N.array2string(lon_array.max()))
#           setattr(f,'geospatial_lon_min',N.array2string(lon_array.min()))
#           setattr(f,'time_coverage_start',str(int(outtime[0]))+" seconds since 1970-01-01 00:00:00.000 UTC")
#           setattr(f,'time_coverage_end',str(int(outtime[-1]))+" seconds since 1970-01-01 00:00:00.000 UTC")
#           setattr(f,'observationDimension','record_length')
#
#           Close netCDF file
#
            f.close()

if __name__=='__main__':
    import sys
    data_dir = sys.argv[1]

    sonic_day(data_dir)
