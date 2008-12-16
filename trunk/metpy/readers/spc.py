from __future__ import division
import urllib2, time, re


UTC = time.gmtime(time.time())
day = UTC[2]
hour = 100 * UTC[3]
minute = UTC[4]
curtime = hour + minute
#curtime = 2100

def spc_parse(product_points):
    point_pairs = ''
    for i in range(0,len(product_points)):
        product_points[i] = product_points[i].strip()
        if re.match(re.compile('\d'), product_points[i]):
            point_pairs += ' '
            point_pairs += product_points[i]
    point_pairs = point_pairs.split()
    lats = []
    lons = []
    for points in point_pairs:
        lats.append(int(points[:4]) / 100)
        tmp_lat =int(points[4:]) / -100
        if tmp_lat > -50:
            lons.append(tmp_lat - 100)
        else:
            lons.append(tmp_lat)
    return lons, lats


def find_valid_time(text):
    for line in text:
        if re.search(re.compile('VALID \d{6,6}Z - \d{6,6}Z'), line):
            line = line.split()
            init_date = int(line[1][:2])
            init_time = int(line[1][2:-1])
            end_date = int(line[3][:2])
            end_time = int(line[3][2:-1])
    return init_date, init_time, end_date, end_time


def is_valid(times):
    init_date = times[0]
    init_time = times[1]
    end_date = times[2]
    end_time = times[3]
    # Handles scenario when everything is on the same day.
    if end_date == day == init_date:
        if init_time <= curtime < end_time:
            return True
    # Handles scenario when 'end_date' is after 'init_date' but doesn't
    # handle end of months.
    if end_date > init_date:
        if end_date > day:
            return True
        if end_date == day and curtime < end_time:
            return True
    # Handles scenario when 'end_date' is a different month
    # than 'init_date'.
    if end_date - init_date < -20:
        if init_date == day and init_time <= curtime:
            return True
        if end_date == day and curtime < end_time:
            return True
        if end_date > day:
            return True


def get_text_snippet(product):
    text = ''
    for i in range(len(product)):
        if re.search(re.compile('AREAS AFFECTED...'), product[i]):
            snippet_begin = i
            break
    total_chars = 0
    for i in range(snippet_begin,len(product)):
        if total_chars < 450 and (450 - len(product[i]) >= 0):
            text += product[i]
            text += '\\n'
            total_chars += len(product[i])
        else:
            break
    return text


if __name__ == '__main__':
    
    from BeautifulSoup import BeautifulSoup as bs
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt


    
    totallons = []
    totallats = []
    text_snippet = []
    valid_threshold = 5
    invalid = 0
    valid = False
    for version in range(0,21):
        URL = 'http://www.srh.noaa.gov/productview.php?pil=SWOMCD&version=%i&max=21' % version
        get_SPC_product = urllib2.urlopen(URL)
        SPC_html = bs(get_SPC_product)
        get_SPC_product.close()
        for html in SPC_html('pre'):
            SPC_product = html.contents[:]
        product = SPC_product[0].split('LAT...LON')
        product_text = product[0].split('\n')
        if is_valid(find_valid_time(product_text)):
            valid = True
            text_snippet.append(get_text_snippet(product_text))
            for i in range(1,len(product)):
                product[i] = product[i].split('\n')
                lons, lats = spc_parse(product[i])
            totallons.append(lons)
            totallats.append(lats)
        else:
            invalid += 1
            if invalid >= valid_threshold:
                break
    map = Basemap(resolution='i',projection='lcc',llcrnrlon=-125,llcrnrlat=23,urcrnrlon=-65,urcrnrlat=50,lat_1=25,lon_0=-100,area_thresh=2500.)
    map.bluemarble()
    map.drawcoastlines(color='white')
    map.drawcountries(linewidth=0.5, color='white')
    map.drawstates(linewidth=0.25, color='white')
    if valid:
        for i in range(len(totallons)):
            X, Y = map(totallons[i], totallats[i])
            plt.plot(X, Y, color='red')
            plt.fill(X, Y, edgecolor='red', facecolor='red', linewidth=1, alpha=.2)
        plt.title('Valid SPC Mesoscale Discussions\n%04iZ' % curtime)
        plt.show()
    else:
        X, Y = map(-98, 40)
        plt.text(X, Y, 'No Valid Mesoscale Discussions', color='red', \
                 fontsize=18, horizontalalignment='center', \
                 verticalalignment='center')
        plt.title('Valid SPC Mesoscale Discussions\n%04iZ' % curtime)
        try:
            plt.show()
        except:
            plt.savefig('valid_mds.png')
