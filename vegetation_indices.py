# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 14:15:19 2025

@author: Robotics
"""
import numpy as np


def safe_divide(numerator, denominator):
    """Divide arrays safely: returns NaN when denominator is zero or invalid."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = 0 # replace inf and -inf with nan
    return result

def ndvi(b,g,r,nir,re):
    return safe_divide(nir - r, nir + r)

def gndvi(b,g,r,nir,re):
    return safe_divide(nir - g, nir + g)

def ndre(b,g,r,nir,re):
    return safe_divide(nir - re, nir + re)

def sipi(b,g,r,nir,re):
    return safe_divide(nir - b, nir + b)

def ngbdi(b,g,r,nir,re):
    return safe_divide(g - b, g + b)

def ngrdi(b,g,r,nir,re):
    return safe_divide(g - r, g + r)

def grdi(b,g,r,nir,re):
    return g - r

def nbgvi(b,g,r,nir,re):
    return safe_divide(b - g, b + g)

def negi(b,g,r,nir,re):
    return safe_divide(2*g - r - b, 2*g + r + b)

def mgrvi(b,g,r,nir,re):
    return safe_divide(g**2 - r**2, g**2 + r**2)

def mvari(b,g,r,nir,re):
    return safe_divide(g - b, g + r - b)

def rgbvi(b,g,r,nir,re):
    return safe_divide(g**2 - b*r, g**2 + b*r)

def tgi(b,g,r,nir,re):
    return g - 0.39*r - 0.61*b

def vari(b,g,r,nir,re):
    return safe_divide(g - r, g + r - b)

def grri(b,g,r,nir,re):
    return safe_divide(g, r)

def nri(b,g,r,nir,re):
    return safe_divide(r, r + g + b)    

def grvi(b,g,r,nir,re):
    return safe_divide(g - r, g + r - b)

def sr(b,g,r,nir,re):
    return safe_divide(nir, r)

def savi(b,g,r,nir,re):
    return safe_divide(1.5*(nir - r), nir + r + 0.5)

def cl(b,g,r,nir,re):
    return safe_divide(nir, re) - 1

def psri(b,g,r,nir,re):
    return safe_divide(r - g, re)

def m3cl(b,g,r,nir,re):
    return safe_divide(nir + r + re, nir - r + re)

def si(b,g,r,nir,re):
    return (r + g + b)/3

def msr(b,g,r,nir,re):
    return safe_divide(safe_divide(nir, r) - 1, safe_divide(nir, r) + 1)

def osavi(b,g,r,nir,re):
    return safe_divide(1.16*(nir - r), nir + r + 0.16)

def rvi(b,g,r,nir,re):
    return safe_divide(nir, r)

def rvi2(b,g,r,nir,re):
    return safe_divide(nir, g)

def tvi(b,g,r,nir,re):
    return 60*(nir - g) - 100*(g - r)

def evi(b,g,r,nir,re):
    return safe_divide(2.5*(nir - r), nir + 6*r - 7.5*b + 1)

def gi(b,g,r,nir,re):
    return safe_divide(g, r)

def tcari(b,g,r,nir,re):
    return 3*((re - r) - 0.2*(re - g)*safe_divide(re, r))

def srpi(b,g,r,nir,re):
    return safe_divide(b, r)

def npci(b,g,r,nir,re):
    return safe_divide(r - b, r + b)

def ndvigb(b,g,r,nir,re):
    return safe_divide(g - b, g + b)

def psri2(b,g,r,nir,re):
    return safe_divide(b - r, g)

def cive(b,g,r,nir,re):
    return 0.44*r - 0.81*g + 0.39*b + 18.79

def nirv(b,g,r,nir,re):
    return nir * ndvi(b,g,r,nir,re)

def dvi(b,g,r,nir,re):
    return nir - r

def msavi(b,g,r,nir,re):
    return safe_divide((2*nir + 1) - np.sqrt((2*nir + 1)**2 - 8*(nir - r)), 2)

def cari(b,g,r,nir,re):
    return re - r - 0.2*(re - g)

def remsr(b,g,r,nir,re):
    return safe_divide(safe_divide(nir, re) - 1, safe_divide(np.sqrt(nir), re) + 1)

def rendvi(b,g,r,nir,re):
    return safe_divide(nir - re, nir + re)

def lci(b,g,r,nir,re):
    return safe_divide(nir - re, nir + r)