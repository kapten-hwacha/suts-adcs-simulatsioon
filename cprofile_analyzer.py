import pstats
p = pstats.Stats('profile_output.prof')
p.strip_dirs().sort_stats('cumulative').print_stats(20)
