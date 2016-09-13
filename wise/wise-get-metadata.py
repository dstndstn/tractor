from __future__ import print_function
import os
import urllib
import urllib2
import threading
#from xml.dom import minidom, Node
import xml.etree.ElementTree as ET

import numpy as np
from astrometry.util.fits import *
from astrometry.util.file import *

def getcat(cat, cols, NC):
    ffn = '%s.fits' % cat
    if os.path.exists(ffn):
        print('File exists:', ffn)
        return

    catfn = '%s.cat.xml' % cat

    if cols is None:
        fn = '%s.dd.xml' % cat
        if not os.path.exists(fn):
            url = 'http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?mode=xml&short=0&catalog=%s' % cat
            print('Retrieving', url)
            f = urllib2.urlopen(url)
            data = f.read()
            f = open(fn, 'w')
            f.write(data)
            f.close()
            print('Wrote', len(data), 'to', fn)
    
        tree = ET.parse(fn)
        root = tree.getroot()
        print('Parsed XML:', root)
    
        colnames = []
        for col in root.iter('column'):
            name = col.find('colname')
            # print '  column:', name.text
            colnames.append(name.text)
    else:
        colnames = [c for c in cols]

    print('Columns:', colnames)

    colnames.remove('cntr')

    fitsfns = []

    i = 0
    while True:
        cn = colnames[:NC]
        colnames = colnames[NC:]
        if len(cn) == 0:
            break
        catfn2 = catfn + '-%i' % i
        ffn2 = '%s-%i.fits' % (cat, i)
        i += 1

        if '4band_p1bs' in cat:
            # Keep at most one "l0file" column per chunk (they fail otherwise)
            cnorig = cn + colnames
            cn = []
            for ii in range(NC):
                c = cnorig.pop(0)
                cn.append(c)
                if 'l0file' in c:
                    break
                if len(cnorig) == 0:
                    break
            colnames = cnorig
            print('Taking columns:', cn)
            print('Leaving columns:', colnames)

        if os.path.exists(ffn2):
            print('File exists:', ffn2)
            fitsfns.append(ffn2)
            continue
        
        cn += ['cntr']

        if (os.path.exists(catfn2) and file_size(catfn2) > 100):
            print('File exists:', catfn2)
        else:
            url2 = 'http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=3&spatial=NONE&catalog=%s&selcols=%s' % (cat, ','.join(cn))
            print('Retrieving', url2)
            cmd = 'wget -nv --timeout 0 -O %s "%s"' % (catfn2, url2)
            print('Running:', cmd)
            os.system(cmd)

        print('Parsing', catfn2)
        tree = ET.parse(catfn2)
        root = tree.getroot()
        print('Parsed XML:', tree)

        # print 'table:'
        # for x in root.iter('table'):
        #     print x
        # print 'TABLE:'
        # for x in root.iter('TABLE'):
        #     print x
        # for x in root:
        #     print 'Root child:', x.tag
    
        tab = list(root.iter('TABLE'))[0]
        fieldnames = []
        fieldtypes = []
        typemap = dict(char=str, int=np.int64, double=np.float64, float=np.float64)
        nullmap = {str:'', np.int64:-1, np.float64:np.nan}
        for f in tab.findall('FIELD'):
            print('Field', f.attrib)
            nm = f.attrib['name']
            ty = f.attrib['datatype']
            fieldnames.append(nm)
            fieldtypes.append(typemap[ty])
        
        data = [[] for f in fieldnames]
        
        datanode = list(root.iter('TABLEDATA'))[0]
        for irow,tr in enumerate(datanode):
            assert(len(tr) == len(fieldtypes))
            try:
                for td,typ,dd in zip(tr, fieldtypes, data):
                    if td.text is None:
                        dd.append(nullmap[typ])
                    else:
                        dd.append(typ(td.text))
            except:
                print('Error in TR row', irow, 'of file', catfn2)
                print(ET.dump(tr))
                raise
        
        data = [np.array(dd, dtype=tt) for dd,tt in zip(data, fieldtypes)]

        del td
        del tr
        del datanode
        del f
        del tab
        del root
        del tree
        
        T = tabledata()
        for dd,nn in zip(data, fieldnames):
            T.set(nn, dd)
        T.about()

        T.writeto(ffn2)
        print('Wrote', ffn2)

        fitsfns.append(ffn2)

    Tall = None
    for fn in fitsfns:
        T = fits_table(fn)
        print('Read', len(T), 'from', ffn2)
        T.about()
        if Tall is None:
            Tall = T
            continue
        assert(np.all(Tall.cntr == T.cntr))
        for c in T.get_columns():
            if c == 'cntr':
                continue
            print('copying column', c)
            Tall.set(c, T.get(c))

    print('Writing:')
    Tall.about()
    Tall.writeto(ffn)
    print('Wrote', ffn)

    
threads = []


if False:
    cols = ['coadd_id', 'ra', 'dec', 'crota', 'moon_lev', 'w1moonrej', 'w2moonrej', 'w3moonrej', 'w4moonrej', 'w1numfrms', 'w2numfrms', 'w3numfrms', 'w4numfrms', 'naxis1', 'naxis2', 'crpix1', 'crpix2', 'ctype1', 'ctype2', 'bunit', 'cdelt1', 'cdelt2', 'mergetype', 'w1magzp', 'w2magzp', 'w3magzp', 'w4magzp', 'w1magzpunc', 'w2magzpunc', 'w3magzpunc', 'w4magzpunc', 'qual_coadd', 'qc_fact', 'qi_fact', 'qa_fact', 'cntr']
    NC = 100
    for cat in ['wise_allsky_4band_p3as_cdd',
                'wise_allsky_3band_p3as_cdd',
                # 'wise_allsky_2band_p3as_cdd', No such table
                ]:
        t = threading.Thread(target=getcat, args=(cat, cols, NC))
        t.start()
        threads.append(t)
        #getcat(cat,cols,NC)

if True:
    cols = None
    #NC = 5
    NC = 1
    for cat in [
        'wise_allsky_2band_p1bs_frm',
        #'wise_allsky_3band_p1bs_frm',
        #'wise_allsky_4band_p1bs_frm',
        ]:
        t = threading.Thread(target=getcat, args=(cat, cols, NC))
        t.start()
        threads.append(t)
    


for t in threads:
    print('Waiting for threads to finish...')
    t.join()
print('Done!')
