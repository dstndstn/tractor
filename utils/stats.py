import pstats

if __name__ == '__main__':
    '''
    Print some stats from python CPU profiling data files
    '''
    import optparse
    import sys
    parser = optparse.OptionParser(usage='%prog <profile.dat>')
    opt,args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)

    fn = args[0]
    st = pstats.Stats(fn)
    st = st.strip_dirs()
    st = st.sort_stats('cumulative')
    st.print_stats()
