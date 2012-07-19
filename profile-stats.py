import sys
import pstats
import numpy as np

def main():
    import optparse
    parser = optparse.OptionParser('%prog: [opts] <profile>...')
    opt,args = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        sys.exit(0)


    for fn in args:

        print
        print 'Cumulative time'
        print

        p = pstats.Stats(fn)
        p.sort_stats('cumulative').print_stats(40)

        p.print_callees(40)

        print
        print 'Time'
        print

        p = pstats.Stats(fn)
        p.sort_stats('time').print_stats(40)
        p.print_callees()


        width,lst = p.get_print_list([40])
        #print 'lst', lst
        if lst:
            p.calc_callees()
            name_size = width
            arrow = '->'
            print 'lst length:', len(lst)
            for func in lst:
                #print 'func', func
                if func in p.all_callees:
                    p.print_call_heading(width, "called...")
                    print pstats.func_std_string(func).ljust(name_size) + arrow,
                    print
                    #p.print_call_line(width, func, p.all_callees[func])
                    cc = p.all_callees[func]
                    #print 'Callees:', cc
                    TT = []
                    CT = []
                    for func,value in cc.items():
                        #print 'func,value', func, value
                        if isinstance(value, tuple):
                            nc, ccx, tt, ct = value
                            TT.append(tt)
                            CT.append(ct)
                            #print '  ', func, ct, tt
                        else:
                            print 'NON-TUPLE', value

                    I = np.argsort(CT)
                    FV = list(cc.items())
                    for i in reversed(I[-40:]):
                        func,value = FV[i]
                        name = pstats.func_std_string(func)
                        if isinstance(value, tuple):
                            nc, ccx, tt, ct = value
                            if nc != ccx:
                                substats = '%d/%d' % (nc, ccx)
                            else:
                                substats = '%d' % (nc,)
                            substats = '%-20s %s %s  %s' % (substats, pstats.f8(tt), pstats.f8(ct), name)
                        print '   ' + substats

                else:
                    p.print_call_line(width, func, {})
                print
                print



if __name__ == '__main__':
    main()


                                                                        
