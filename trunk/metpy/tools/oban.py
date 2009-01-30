from itertools import izip
import numpy as N
from constants import *

def rms(diffs):
  return N.sqrt(N.average(diffs**2))

def grid_point_dists(grid_x, grid_y, ob_x, ob_y):
  "Calculates distances for each grid point to every ob point"
  return N.hypot(grid_x[...,N.newaxis] - ob_x[N.newaxis,N.newaxis,...],
    grid_y[...,N.newaxis] - ob_y[N.newaxis,N.newaxis,...])

def adjust_field(field, xgrad, ygrad, grid_x, grid_y, ob_x, ob_y):
  '''Makes a 3D field with the data adjusted by the gradient to each grid point
    from every observation point'''
  return field + xgrad * (grid_x[...,N.newaxis] - ob_x[N.newaxis,...])\
    + ygrad * (grid_y[...,N.newaxis] - ob_y[N.newaxis,...])

def analyze_grid_multipass(ob_data, grid_x, grid_y, ob_x, ob_y, num_passes,
  weight_func, params, background = None):
  '''Calculate a value at each grid point using multiple passes of an objective
  analysis technique'''
  if background is None:
    mod_param = (params[0], 1.0)
    background = analyze_grid(ob_data, grid_x, grid_y, ob_x, ob_y, weight_func,
      mod_param)
    num_passes -= 1
  for i in range(num_passes):
    ob_incs = get_ob_incs(ob_x, ob_y, ob_data, grid_x[0], grid_y[:,0],
      background)
    print 'pass: %d rms: %f' % (i, rms(ob_incs))
    background = analyze_grid(ob_incs, grid_x, grid_y, ob_x, ob_y,
      weight_func, params) + background
  return background
  
def analyze_grid(ob_data, grid_x, grid_y, ob_x, ob_y, weight_func, params):
  '''Calculates a value at each grid point based on the observed data in
    ob_data.  grid_point_dists is a 3D array containing the distance for each
    grid point to every observation'''
  try:
    params[0]
  except TypeError:
    params = (params,)
  weights = weight_func(grid_point_dists(grid_x, grid_y, ob_x, ob_y), *params)
  final = (weights * ob_data).sum(axis=2)/weights.sum(axis=2)
  try:
    final[N.isnan(final)] = 0.0
  except:
    pass
  return final

def uniform_weights(dists, radius):
  weights = N.ones_like(dists)
  weights[dists > radius] = 0.0
  return weights

def cressman_weights(dists, radius):
  dist_sq = dists * dists
  rad_sq = radius * radius
  weights = (rad_sq - dist_sq)/(rad_sq + dist_sq)
  weights[dists > radius] = 0.0
  return weights
  
def barnes_weights(dists, kappa0, gamma):
  weights = N.exp(-dists**2 / (kappa0 * gamma))
  critical_radius = N.sqrt(15 * kappa0)
  weights[dists > critical_radius] = 0.0
  return weights

def bilinear(x, y, data, xloc, yloc):
  xind = find_axis_index(x, xloc)
  yind = find_axis_index(y, yloc)
  xw = (xloc - x[xind])/(x[xind+1] - x[xind])
  x_weights = N.array([1-xw, xw])
  yw = (yloc - y[yind])/(y[yind+1] - y[yind])
  y_weights = N.array([1-yw, yw])
  return N.dot(y_weights, N.dot(data[yind:yind+2,xind:xind+2], x_weights))

def find_axis_index(axis_vals, location):
  if location > axis_vals[-1] or location < axis_vals[0]:
    raise ValueError, "Location out of bounds"
  for ind,val in enumerate(axis_vals):
    if location < val:
      break
  return ind - 1

def get_wind_comps(spd, dir):
  u = -spd * N.sin(dir * rad_per_deg)
  v = -spd * N.cos(dir * rad_per_deg)
  return u, v

def get_ob_incs(obx, oby, ob, grid_x, grid_y, field, cressman_radius = None):
  ob_inc = list()
  mask = N.zeros(ob.size)
  for x,y,value in izip(obx,oby,ob):
    try:
      interp_val = bilinear(grid_x, grid_y, field, x, y)
      ob_inc.append(value - interp_val)
    except ValueError:
      if cressman_radius is None:
        mask[len(ob_inc)] = 1
        ob_inc.append(0.0)
      else:
      #Ugly hack here to allow the one station off the grid to be interpolated
        xg,yg = N.meshgrid(grid_x, grid_y)
        interp_val = analyze_grid(field.flatten(), N.array(x, ndmin = 2),
          N.array(y, ndmin = 2), xg.flatten(), yg.flatten(),
          cressman_weights, cressman_radius)
        if N.isnan(interp_val):
          interp_val = value
#        mask[len(ob_inc) - 1] = 1
        ob_inc.append(value - interp_val.flatten()[0])
  return N.ma.array(ob_inc, mask = mask)

def calc_barnes_param(spacing):
  '''Calculate the Barnes analysis smoothing parameter, kappa0, from the
  average grid spacing'''
  return 5.052 * (2.0 * spacing / N.pi)**2
