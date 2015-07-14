#!/usr/bin/env python
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------#
#TODO check the importings
#TODO check the paradigm to choose
#TODO check the iraf wrapping syntax
#TODO check the parameters inherent to EABA
#TODO caracterizar calidad de los flats; hacer esto automaticamente
#TODO check if we can use PyDS9, this would be much better
#------------------------------------------------------------------------------#
import glob
import os
import sys
import shutil,shlex,subprocess
import fnmatch
#------------------------------------------------------------------------------#
#import pdb
#import numpy as np
from astropy.io import fits
from pyraf import iraf
#------------------------------------------------------------------------------#
from iraf import noao,imred, ccdred
from iraf import  ccdhedit
#------------------------------------------------------------------------------#
#from AstroObject.image import ImageStack
#from AstroObject.iraftools import UseIRAFTools

#------------------------------------------------------------------------------#
def load_ccdproc():
    iraf.unlearn("ccdproc")
    #---------------------------------------------------------------------------#
    iraf.ccdproc.ccdtype = ''
    iraf.ccdproc.oversca = 'no'
    iraf.ccdproc.trim = 'no'
    iraf.ccdproc.zerocor = 'yes'
    iraf.ccdproc.darkcor = 'no'
    iraf.ccdproc.fixpix = 'no'
    iraf.ccdproc.flatcor = 'no'
    iraf.ccdproc.illumcor = 'no'
    iraf.ccdproc.fringec = 'no'
    iraf.ccdproc.readcor = 'no'
    iraf.ccdproc.scancor = 'no'
    iraf.ccdproc.interac = 'no'
    iraf.ccdproc.biassec = ''#biassection
    iraf.ccdproc.trimsec = ''#trimsection
#------------------------------------------------------------------------------#
def write_tolist(lista,filename,directory=None,subdir=False):#, workdir):
    """function that generates a file list for iraf task executions"""
    f = open(str(filename), 'w+')
    for i in range(len(lista)):
        if subdir:
            mkdir(directory)
            copy_to_dir(os.path.abspath(lista[i]), directory)#os.path.join(workdir,directory))
        #nfile = os.path.abspath(directory+lista[i])
        nfile = lista[i]
        f.write(str(nfile)+'\n')
    f.close()
    return(os.path.abspath(str(filename)))
#------------------------------------------------------------------------------#
def mkdir(dirname):
    """Creates a directory if it didn't exist before"""
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
#------------------------------------------------------------------------------#
def cd(target):
    """Changes to a directory, pyraf included"""
    os.chdir(target)
    iraf.cd(target)
#------------------------------------------------------------------------------#
def copy_to_dir(orig, target):
    """Copy files to a diferent directory"""
    filename = os.path.basename(orig)
    filepath = os.path.abspath(orig)
    target = os.path.abspath(target)
    target = os.path.join(target,filename)
    #print 'file {} to target {}'.format(filepath,target)
    shutil.copyfile(filepath, target)
#------------------------------------------------------------------------------#
#%#--------------------------------------------------------------------------#%%
#------------------------------------------------------------------------------#
def preprocess(path): #, mode='EABA'):
    """ Pipeline that applies the basic CCD reduction tasks.
    Asks for the path were all files are present, and
    performs Bias combination, Dark combination, Flat
    combination and the inherent substractions
    Works in TORITOS mode, and EABA mode, the latter using UBVRI
    filters.

    Pipeline que aplica una reducción básica a imágenes de tipo CCD
    Realiza combinación de Bias, de Darks, y Flats.
    Luego realiza las restas inherentes
    Funciona en dos modos, TORITOS y EABA, el último usando filtros UBVRI
    """
    bornpath = os.path.abspath('.')
    workdir = os.path.abspath(path)
    biasdir=os.path.join(workdir,"bias")
    flatdir=os.path.join(workdir,"flat")
    darkdir=os.path.join(workdir,"dark")
    scidir=os.path.join(workdir,"science")
    for x in [biasdir,flatdir,darkdir,scidir]:
        print "creating {} directory".format(str(x))
        mkdir(x)
    cd(workdir)
    #-------------------------------------------------------------------------#
    fitslist = []
    for root,dirs,files in os.walk(path, topdown=True):
        for name in fnmatch.filter(files, '*.fit*'):
            fitslist.append(os.path.join(root,name))
    #fitslist = glob.glob('*.fits') + glob.glob('*.fit')
    #load_ccdred()
    #-------------------------------------------------------------------------#
    imagetypes={}
    for img in fitslist:
        imghead = fits.getheader(img)
        imgty = imghead['IMAGETYP']
        imagetypes[str(img)]=imgty
    #-------------------------------------------------------------------------#
    # now we get the image types in the dictionary imagetypes
    # now we make different lists for different image types
    #-------------------------------------------------------------------------#
    biaslist = []
    sciencelist = []
    flatlist = []
    darklist =[]
    #-------------------------------------------------------------------------#
    for k,v in imagetypes.iteritems():
        print k,v
        if v.upper()=='LIGHT' or v.upper()=='OBJECT' or v.upper()=='LIGHT FRAME':
    #            print str(k)+' is a '+str(v)+' file'
            sciencelist.append(str(k))
        elif v.upper()=='BIAS' or v.upper()=='ZERO' or v.upper()=='BIAS FRAME':
    #            print str(k)+' is a '+str(v)+' file'
            biaslist.append(str(k))
        elif v.upper()=='FLAT' or v.upper()=='FLAT FRAME' or v.upper()=='FLAT FIELD':
    #           print str(k)+' is a '+str(v)+' file'
            flatlist.append(str(k))
        elif v.upper()=='DARK' or v.upper()=='DARK FRAME':
    #           print str(k)+' is a '+str(v)+' file'
            darklist.append(str(k))
    #-------------------------------------------------------------------------#
    insp = raw_input("Inspeccionar imagenes de correccion con ds9? (si/NO)\n")
    if insp.upper() in ('S', 'Y', 'SI', 'YES'):
        print('\n')
        print(u"""Comienza la inspeccion visual de bias
            ante cualquier anomalia en alguna imagen se la vetará
              Starting visual inspection of Bias frames. If there exists any
              anomalies image will be discarded""")
        for idx,img in enumerate(list(biaslist)):
            I = raw_input("inspeccionar {} mediante ds9? (si/NO)\n".format(str(img)))
            if I.upper() in ('S', 'Y','SI','YES'):
                print(""" inspeccionando {} mediante ds9""".format(str(img)))
                print("\n")
                command = shlex.split('ds9 {}'.format(str(img)))
                subprocess.call(command)
                V = raw_input(" Vetar la imagen? (si/NO)")
                print("\n")
                if V.upper() in ('S', 'Y','SI','YES'):
                    S = raw_input('Es una imagen de ciencia? (LIGHT)  (SI/no)')
                    print("\n")
                    if S.upper() in ('S', 'Y','SI','YES'):
                        hdu = fits.open(img, mode='update')
                        hdr = hdu[0].header
                        hdr.set('IMAGETYP','LIGHT')
                        hdu.close()
                        sciencelist.append(img)
                    elif S.upper() in ('N', 'NO'):
                        os.rename(img, img+'.vet')
                    biaslist.remove(img)
        #-------------------------------------------------------------------------#
        print('\n')
        print(u"""Comienza la inspeccion visual de flats
            ante cualquier anomalia en alguna imagen se la vetará \n""")
        for idx,img in enumerate(list(flatlist)):
            I = raw_input("inspeccionar {} mediante ds9? (si/NO)\n".format(str(img)))
            if I.upper() in ('S', 'Y','SI','YES'):
                print(""" inspeccionando {} mediante ds9""".format(str(img)))
                print("\n")
                command = shlex.split('ds9 {}'.format(str(img)))
                subprocess.call(command)
                V = raw_input(" Vetar la imagen? (si/NO) ")
                print("\n")
                if V.upper() in ('S', 'Y','SI','YES'):
                    S = raw_input('Es una imagen de ciencia? (LIGHT)  (SI/no) ')
                    print("\n")
                    if S.upper() in ('S', 'Y','SI','YES'):
                        hdu = fits.open(img, mode='update')
                        hdr = hdu[0].header
                        hdr.set('IMAGETYP','LIGHT')
                        hdu.close()
                        sciencelist.append(img)
                    elif S.upper() in ('N', 'NO'):
                        os.rename(img, img+'.vet')
                    flatlist.remove(img)
        #-------------------------------------------------------------------------#
        print("\n")
        print(u"""Comienza la inspeccion visual de darks
            ante cualquier anomalia en alguna imagen se la vetará \n""")
        for idx,img in enumerate(list(darklist)):
            I = raw_input("inspeccionar {} mediante ds9? (si/NO)\n".format(str(img)))
            if I.upper() in ('S', 'Y','SI','YES'):
                print(""" inspeccionando {} mediante ds9""".format(str(img)))
                print("\n")
                command = shlex.split('ds9 {}'.format(str(img)))
                subprocess.call(command)
                V = raw_input(" Vetar la imagen? (si/NO)")
                print("\n")
                if V.upper() in ('S', 'Y','SI','YES'):
                    S = raw_input('Es una imagen de ciencia? (LIGHT)  (SI/no)')
                    print("\n")
                    if S.upper() in ('S', 'Y','SI','YES'):
                        hdu = fits.open(img, mode='update')
                        hdr = hdu[0].header
                        hdr.set('IMAGETYP','LIGHT')
                        hdu.close()
                        sciencelist.append(img)
                    elif S.upper() in ('N', 'NO'):
                        os.rename(img, img+'.vet')
                    darklist.remove(img)
    #-------------------------------------------------------------------------#
    #posee listas de todos los files.
    #comienzo por los bias:
    #primero creo una lista (file) para darle a zerocombine
    write_tolist(biaslist, 'lbias', biasdir)#, workdir)
    #-------------------------------------------------------------------------#
    iraf.ccdhedit('@lbias', parameter='imagetype', value='zero')
    iraf.zerocombine.unlearn()
    iraf.zerocombine('@lbias',output='Zero.fits')
    #baseflat = [os.path.basename(x) for x  in flatlist]
    #basesci  = [os.path.basename(x) for x  in sciencelist]
    #basedark = [os.path.basename(x) for x  in darklist]
    #-------------------------------------------------------------------------#
    #ahora corrijo las imagenes de flat, dark y objetos por bias.
    load_ccdproc()
    iraf.ccdproc.zero = 'Zero.fits'
    for names in flatlist:
        iraf.ccdproc(names, output=os.path.join(flatdir,'fz'+os.path.basename(names)))
    for names in darklist:
        iraf.ccdproc(names, output=os.path.join(darkdir,'dz'+os.path.basename(names)))
    #-------------------------------------------------------------------------#
    #recreate lists of new corrected objects
    flatlist = []
    darklist = []
    for root,dirs,files in os.walk(path, topdown=True):
        for name in fnmatch.filter(files, 'fz*'):
            flatlist.append(os.path.join(flatdir,name))
    for root,dirs,files in os.walk(path, topdown=True):
        for name in fnmatch.filter(files, 'dz*'):
            darklist.append(os.path.join(darkdir,name))
    #-------------------------------------------------------------------------#
    #combino darks tal como hice con los bias
    #-------------------------------------------------------------------------#
    write_tolist(darklist, 'ldark')
    #write_tolist(sciencelist, 'lsci')
    iraf.darkcombine.unlearn()
    iraf.ccdhedit('@ldark', parameter='imagetype',value='dark')
    iraf.darkcombine('@ldark',output='Dark.fits')
    #-------------------------------------------------------------------------#
    # discrimino por filtro los datos. es importante esta parte!
    #.------------------------------------------------------------------------#
    #SE USARON FILTROS?
    filteruse = True
    for v in flatlist:
        try:
            hdr=fits.getheader(v)
            FILTER = hdr['FILTER']
            print 'FILTROS HALLADOS'
        except KeyError:
            filteruse=False
            print 'NO SE USARON FILTROS'
    #---------------------------IF FILTERS USED ------------------------------#
    if filteruse:
        FD = {'U':[], 'B':[], 'V':[], 'R':[], 'I':[]}
        for idx, img in enumerate(list(flatlist)):
            hdr = fits.getheader(img)
            FR = hdr['FILTER']
            #creo un diccionario de filtros
            FD[str(FR)].append(img)
        print(FD)
        #-------------------------------------------------------------------------#
        #combino los flats
        #-------------------------------------------------------------------------#
        for k,v in FD.iteritems():
            if not v==[]:
                print('writing to list for {} filter'.format(k))
                print(v)
                lname = str(k)+'flat'
                #print(lname)
                write_tolist(v, lname)
                iraf.flatcombine.unlearn()
                iraf.flatcombine.combine='median'
                iraf.ccdhedit('@'+lname, parameter='imagetype', value='flat')
                iraf.flatcombine('@'+lname, output='Flat'+k+'.fits')
        #-------------------------------------------------------------------------#
        SD = {'U':[], 'B':[], 'V':[], 'R':[], 'I':[]}
        for idx, img in enumerate(list(sciencelist)):
            hdr = fits.getheader(img)
            FR = hdr['FILTER']
            #creo un diccionario de filtros
            SD[str(FR)].append(img)
        print(SD)
        #-------------------------------------------------------------------------#
        for k,v in SD.iteritems():
            iraf.ccdproc.flatcor = 'yes'
            iraf.ccdproc.darkcor = 'yes'
            iraf.ccdproc.dark    = 'Dark.fits'
            for img in v:
                if os.path.isfile('Flat'+k+'.fits'):
                    iraf.ccdproc.flat = 'Flat'+k+'.fits'
                    iraf.ccdproc(img, output = 'reduced_'+img)
    #----------------------IF NO FILTERS--------------------------------------#
    else:
        lname = 'flist'
        write_tolist(flatlist, lname)
        iraf.flatcombine.unlearn()
        iraf.flatcombine.combine='median'
        iraf.ccdhedit('@'+lname, parameter='imagetype', value='flat')
        iraf.flatcombine('@'+lname, output='Flat.fits')
        iraf.ccdproc.flatcor = 'yes'
        iraf.ccdproc.darkcor = 'yes'
        iraf.ccdproc.dark    = 'Dark.fits'
        iraf.ccdproc.flat    = 'Flat.fits'
        iraf.ccdproc.ccdtype = ''
        for sciname in sciencelist:
            print 'ccdproc of ',sciname,'\n',os.path.join(scidir,'reduced_'+os.path.basename(sciname))
            iraf.ccdproc(sciname, output=os.path.join(scidir,\
                            'reduced_'+os.path.basename(sciname)))
    #[os.remove(x) for x in fitslist]
    aux = glob.glob('sz*.fits')+glob.glob('dz*.fits')+glob.glob('fz*.fits')
    [os.remove(x) for x in aux]
    aux = glob.glob('reduced*.fits')
    [os.rename(x, x[10:-4].upper()+x[-4:].lower()) for x in aux]
    #-------------------------------------------------------------------------#
    print(u""" Imágenes reducidas exitosamente APARENTEMENTE,
            chequea todo obsesivamente, porque ni el desarrollador de
            esto confia en que ande bien \n
            """)
    #-------------------------------------------------------------------------#
    cd(bornpath)
    #print sys.argv[:]
    return()

#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    path = sys.argv[1]
    preprocess(path)
