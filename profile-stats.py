import sys
import pstats

def main():
	import optparse
	parser = optparse.OptionParser('%prog: [opts] <profile>...')
	opt,args = parser.parse_args()
	if len(args) == 0:
		parser.print_help()
		sys.exit(0)

	for fn in args:
		p = pstats.Stats(fn)
		p.sort_stats('cumulative').print_stats(40)
		#p.strip_dirs().sort_stats(-1).print_stats()


if __name__ == '__main__':
	main()
