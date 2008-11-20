__all__ = ['meteogram']

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def meteogram(site, dt=None, **fields):
    if dt is None:
        import datetime
        dt = datetime.date.today()

    time = fields['time']/60.
    temp = fields['temp']
    relh = fields['relh']
    press = fields['press']
    wspd = fields['wspd']
    
    ax1 = plt.subplot(3,1,1)
    plt.plot(time, relh, 'g')
    plt.ylabel('Rel. Humidity (%)')
    plt.xlim(0,24)
    plt.title('Meteogram for %s on %s' % (site.upper(), dt))
    ax1.xaxis.set_major_formatter(NullFormatter())
    
    ax2 = plt.subplot(3,1,2)
    plt.plot(time, temp, 'r')
    plt.ylabel('Temperature (C)')
    plt.xlim(0,24)
    ax2.xaxis.set_major_formatter(NullFormatter())
    
    ax3 = plt.subplot(3,1,3)
    plt.plot(time, press, 'k')
    plt.xlabel('Time (hour)')
    plt.ylabel('Pressure (mb)')
    ax32 = plt.twinx(ax3)
    ax32.plot(time, wspd, 'b')
    ax32.set_ylabel('Wind Speed (m/s)')
    plt.xlim(0,24)
    plt.xticks(range(0,25,3))
    
    plt.subplots_adjust(hspace=0.1)
    
    return [ax1, ax2, (ax3, ax32)]
