from cProfile import Profile
from pstats import Stats

from metpy.mapping.points import interp_points
from metpy.mapping.tests.test_MPGridding import create_test_data

def run_profile():

	prof = Profile()

	x, y, t = create_test_data("air_temperature", "+init=EPSG:4326", '+init=EPSG:5069')

	prof.enable()

	interp_points(x, y, t, "natural_neighbor", 100000, 100000)
	interp_points(x, y, t, "natural_neighbor", 50000, 50000)
	interp_points(x, y, t, "natural_neighbor", 25000, 25000)
	
	prof.disable()

	prof.dump_stats('mystats.stats')
	with open('mystats_output.txt', 'wt') as output:
		stats = Stats('mystats.stats', stream=output)
		stats.sort_stats('cumulative', 'time')
		stats.print_stats()