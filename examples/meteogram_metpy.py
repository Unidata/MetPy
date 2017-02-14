
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib as mpl

from metpy.units import units
from metpy.calc import dewpoint_rh

def calc_dewpoint(T,RH):
    RH = np.ma.masked_values(RH,-999.9)
    num = np.log(RH/100) + (17.625*T)/(243.04+T)
    denom = 17.625 - np.log(RH/100) - (17.625*T)/(243.04+T)
    return 243.04*num/denom

def calc_mslp(T,P,h):
    return P*(1-(0.0065*h)/(T+0.0065*h+273.15))**(-5.257)

def C_to_F(temp):
    return temp*1.8 + 32


# Make meteogram plot
class Meteogram(object):
    """ Plot a time series of meteorological data from a particular station as a meteogram with standard variables 
    to visualize, including thermodynamic, kinematic, and pressure. The functions below control the plotting
    of each variable. 
    
    TO DO: Make the subplot creation dynamic so the number of rows is not static as it is currently. """
    def __init__(self,fig,dates,probeid,time=dt.datetime.utcnow(),axis=0):
        """ 
        Required input:
            fig: figure object
            dates: array of dates corresponding to the data
            probeid: ID of the station
        Optional Input:
            time: Time the data is to be plotted
            axis: number that controls the new axis to be plotted (FOR FUTURE)
        """
        self.start = dates[0]
        self.fig = fig
        self.end = dates[-1]
        self.axis_num = 0
        self.dates = mpl.dates.date2num(dates)
        self.time = time.strftime("%Y-%m-%d %H:%M UTC")
        self.title = 'Latest Ob Time: {0}\nProbe ID: {1}'.format(self.time,probeid)

    def plot_winds(self,WS,WD,WSMAX,plot_range=[0,20,1]):
        """ 
        Required input:
            WS: Wind speeds (knots)
            WD: Wind direction (degrees)
            WSMAX: Wind gust (knots)
        Optional Input:
            plot_range: Data range for making figure
        """        
        # PLOT WIND SPEED AND WIND DIRECTION
        self.ax1 = fig.add_subplot(4, 1, 1)
        ln1 = self.ax1.plot(self.dates, 
                   WS, 
                   label='Wind Speed'
                   )
        plt.fill_between(self.dates,WS,0)
        self.ax1.set_xlim(self.start,self.end)
        #self.ax1.get_xaxis().set_ticks([(self.starttime+dt.timedelta(hours=i*4)) for i in np.arange(0,7)])
        plt.ylabel('Wind Speed (knots)', multialignment='center')
        plt.grid(b=True,which='major',axis='y',color='k',linestyle='--',linewidth=0.5)
        ln2 = self.ax1.plot(self.dates,
            WSMAX,
            '.r',
            label='3-sec Wind Speed Max')
        plt.setp(self.ax1.get_xticklabels(), visible=True)
        ax7 = self.ax1.twinx()
        ln3 = ax7.plot(self.dates, 
                   WD, 
                   '.k', linewidth=0.5, label='Wind Direction')
        plt.ylabel('Wind\nDirection\n(degrees)', multialignment='center')
        plt.ylim(0,360)
        plt.yticks(np.arange(45,405,90),['NE','SE','SW','NW'])
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))
        ax7.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5,1.2),ncol=3,prop={'size':12})

        
    def plot_thermo(self,T,TD,plot_range=[10,90,2]): 
        """ 
        Required input:
            T: Temperature (deg F)
            TD: Dewpoint (deg F)
        Optional Input:
            plot_range: Data range for making figure
        """        
        # PLOT TEMPERATURE AND DEWPOINT       
        self.ax2 = fig.add_subplot(4, 1, 2,sharex=self.ax1)
        ln4 = self.ax2.plot(self.dates,
                   T,
                   'r-',
                   label='Temperature')
        plt.fill_between(self.dates,
                     T,
                     TD,
                     color='r')
        plt.setp(self.ax2.get_xticklabels(), visible=True)
        plt.ylabel('Temperature\n(F)', multialignment='center')
        plt.grid(b=True,which='major',axis='y',color='k',linestyle='--',linewidth=0.5)
        self.ax2.set_ylim(plot_range[0],plot_range[1],plot_range[2])
        ln5 = self.ax2.plot(self.dates,
                  TD,
                  'g-',
                   label='Dewpoint')
        plt.fill_between(self.dates,
                     TD,
                     plt.ylim()[0],
                     color='g')
        ax_twin = self.ax2.twinx()
        #    ax_twin.set_ylim(20,90,2)
        ax_twin.set_ylim(plot_range[0],plot_range[1],plot_range[2])
        lns = ln4+ln5
        labs = [l.get_label() for l in lns]
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))

        self.ax2.legend(lns,labs,loc='upper center', bbox_to_anchor=(0.5,1.2),ncol=2,prop={'size':12})
        
    def plot_rh(self,RH,plot_range=[0,100,4]):
        """ 
        Required input:
            RH: Relative humidity (%)
        Optional Input:
            plot_range: Data range for making figure
        """        
        # PLOT RELATIVE HUMIDITY
        self.ax3 = fig.add_subplot(4, 1, 3,sharex=self.ax1)
        self.ax3.plot(self.dates,
             RH,
             'g-',
             label='Relative Humidity')
        self.ax3.legend(loc='upper center',bbox_to_anchor=(0.5,1.22),prop={'size':12})
        plt.setp(self.ax3.get_xticklabels(),visible=True)
        plt.grid(b=True,which='major',axis='y',color='k',linestyle='--',linewidth=0.5)
        self.ax3.set_ylim(plot_range[0],plot_range[1],plot_range[2])
        plt.fill_between(self.dates,RH,plt.ylim()[0],color='g')
        plt.ylabel('Relative Humidity\n(%)',multialignment='center')
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))
        axtwin = self.ax3.twinx()
        axtwin.set_ylim(plot_range[0],plot_range[1],plot_range[2])
        
    def plot_pressure(self,P,plot_range=[970,1030,2]):
        """ 
        Required input:
            P: Mean Sea Level Pressure (hPa)
        Optional Input:
            plot_range: Data range for making figure
        """        
        # PLOT PRESSURE
        self.ax4 = fig.add_subplot(4, 1, 4,sharex=self.ax1)
        self.ax4.plot(self.dates,
             P,'m',label='Mean Sea Level Pressure')
        plt.ylabel('Mean Sea\nLevel Pressure\n(mb)', multialignment='center')
        plt.ylim(plot_range[0],plot_range[1],plot_range[2])
        #    plt.ylim(920,970)
        axtwin = self.ax4.twinx()
        axtwin.set_ylim(plot_range[0],plot_range[1],plot_range[2])
        #    axtwin.set_ylim(920,970)
        plt.fill_between(self.dates,P,plt.ylim()[0],color='m')
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))

        self.ax4.legend(loc='upper center',bbox_to_anchor=(0.5,1.2),prop={'size':12})
        plt.grid(b=True,which='major',axis='y',color='k',linestyle='--',linewidth=0.5)        
        plt.setp(self.ax4.get_xticklabels(), visible=True)
        
        # OTHER OPTIONAL AXES TO PLOT
        # plot_irradiance
        # plot_precipitation
        
    

# set the starttime and endtime for plotting, 24 hour range
endtime      = dt.datetime(2016,3,31,22,0,0,0)
starttime    = endtime - dt.timedelta(hours=24)

# Height of the station to calculate MSLP
hgt_example = 292.


# Parse dates from .csv file, knowing their format as a string and convert to datetime
def parse_date(date):
    return dt.datetime.strptime(date,"%Y-%m-%d %H:%M:%S")

testdata = np.genfromtxt(get_test_data('test.csv', False), names=True, dtype=None,
                         usecols=list(range(1,8)),
                         converters={'DATE': parse_date}, delimiter=',')

# Temporary variables for ease
temp = testdata['T']
pres = testdata['P']
rh = testdata['RH']
ws = testdata['WS']
wsmax = testdata['WSMAX']
wd = testdata['WD']
date = testdata['DATE']    

# ID For Plotting on Meteogram
probe_id = '0102A'

data = dict()
data['wind_speed'] = (np.array(ws)*units('m/s')).to(units('knots'))
data['wind_speed_max'] = (np.array(wsmax)*units('m/s')).to(units('knots'))
data['wind_direction'] = np.array(wd)*units('degrees')
data['dewpoint'] = dewpoint_rh((np.array(temp)*units('degC')).to(units('K')),np.array(rh)/100.).to(units('degF'))
data['air_temperature'] = (np.array(temp)* units('degC')).to(units('degF')) 
data['mean_slp'] = calc_mslp(np.array(temp),np.array(pres),hgt_example) * units('hPa')
data['relative_humidity'] = np.array(rh)
data['times'] = np.array(date)    

fig = plt.figure(figsize=(20, 16))
meteogram = Meteogram(fig,data['times'],probe_id)
meteogram.plot_winds(data['wind_speed'],data['wind_direction'],data['wind_speed_max'])        
meteogram.plot_thermo(data['air_temperature'],data['dewpoint']) 
meteogram.plot_rh(data['relative_humidity']) 
meteogram.plot_pressure(data['mean_slp'])
fig.subplots_adjust(hspace=0.5)
plt.show()
#plt.text(1,1.02,meteogram.title,horizontalalignment='right',verticalalignment='bottom')        



