import os

def get_pt():

    fname = 'imforopt.png'
    pnm = "imforopt.pnm"
    svgname = "imforopt.svg"
    os.system("convert %s %s" % (fname, pnm))
    os.system("potrace -s -o %s %s" % (svgname, pnm))
