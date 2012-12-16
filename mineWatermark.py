__author__ = 'liuc'

import pywt, Image, numpy

def insert(imgIn,watermark_code,level):
    return _inoutwatermark(imgIn,level,1,watermark_code)

def extract(imgIn,level):
    return _inoutwatermark(imgIn,level,0)

def _inoutwatermark(imgIn,level,mode,watermark_code=[0]):

    origImg=imgIn.copy()

    min_size=min(origImg.size)
    imgIn = imgIn.crop([0,0,min_size,min_size])

    img_size = imgIn.size

    imgBandsName = imgIn.getbands()
    imgBands = imgIn.split()

    if imgBandsName==('R','G','B'):
        imgInR = numpy.array(imgBands[0].getdata()).reshape(img_size)
        imgInG = numpy.array(imgBands[1].getdata()).reshape(img_size)
        imgInB = numpy.array(imgBands[2].getdata()).reshape(img_size)

        imgInY = 0.299*imgInR + 0.587*imgInG + 0.114*imgInB
        imgInU = (imgInB - imgInY)*0.565
        imgInV = (imgInR - imgInY)*0.713

        imgBand = imgInY
    else:
        imgBand = numpy.array(imgBands[0].getdata()).reshape(img_size)


    [imgBandNew, code] = _dwt([imgBand],level,watermark_code,mode,0)

    imgBandNew=imgBandNew[0]

    if mode==1:
        if imgBandsName==('R','G','B'):
            imgRGB=[]
            imgRGB.append((imgBandNew + 1.403 * imgInV).reshape((-1,img_size[0]*img_size[1])).tolist()[0])
            imgRGB.append((imgBandNew - 0.344 * imgInU - 0.714 * imgInV).reshape((-1,img_size[0]*img_size[1])).tolist()[0])
            imgRGB.append((imgBandNew + 1.770 * imgInU).reshape((-1,img_size[0]*img_size[1])).tolist()[0])

            imgRGB=[map(int,x) for x in imgRGB]
            imgOut = Image.new("RGB",img_size)
            imgOut.putdata(zip(*imgRGB))

        else:
            imgOut = Image.new("L",img_size)
            imgOut.putdata(imgBandNew.reshape((-1,img_size[0]*img_size[1])).tolist()[0])

        origImg.paste(imgOut,[0,0,min_size,min_size])

        return [origImg, code]
    else:
        return code


def _dwt(imgData,level,watermark_code,mode,filter=0,scale=2):
    # imgData = [ [imgData] , ... ]
    #dwtData =  [[cA, (cH, cV, cD)],....]
    # _cA = _cD = 0 works only with wavelet with length = 2 (haar, db1, ...)

    _cA = 0  # =1 for cA descend!
    _cD = 1 #
    _cHPOS = 0
    _cVPOS = 1
    _cDPOS = 2

    dwtData = [pywt.dwt2(x,'db1') for x in imgData]

    if level==1:

        if len(dwtData)==1:
            dwtMark = [ dwtData[0][1][_cHPOS],dwtData[0][1][_cVPOS] ]
        else:
            dwtMark = [ dwtData[0][1][_cHPOS],dwtData[1][1][_cVPOS] ]

        # Insert now the watermark
        if mode==1:

            [dwtMark, returnCode] = _embed(dwtMark,watermark_code)

            if len(dwtData)==1:
                dwtMark.append(dwtData[0][1][_cDPOS])

                dwtData=[ [dwtData[0][0], tuple(dwtMark)] ]
            else:
                dwtMark1=[dwtMark[0], dwtData[0][1][_cVPOS], dwtData[0][1][_cDPOS]]
                dwtMark2=[dwtData[1][1][_cHPOS], dwtMark[1], dwtData[1][1][_cDPOS]]

                dwtData=[ [dwtData[0][0], tuple(dwtMark1)], [dwtData[1][0], tuple(dwtMark2)] ]
        else:
            returnCode = _extract(dwtMark)

        retImgData = [pywt.idwt2(x,'db1') for x in dwtData]
        return [retImgData, returnCode]

    if (_cA==1):
        dwtMark = [ dwtData[0][0] ]
    elif (_cD==1):
        dwtMark = [ dwtData[0][1][_cDPOS] ]
    else:
        if len(dwtData)==1:
            dwtMark = [ dwtData[0][1][_cHPOS],dwtData[0][1][_cVPOS] ]
        else:
            dwtMark = [ dwtData[0][1][_cHPOS],dwtData[1][1][_cVPOS] ]

    [dwtMark, returnCode] = _dwt(dwtMark,level-1,watermark_code,mode,filter,scale*2)

    if (_cA==1):
        dwtData = [ [dwtMark[0], dwtData[0][1] ] ]
    elif (_cD==1):
        dwtData=[ [dwtData[0][0], tuple([dwtData[0][1][_cHPOS],dwtData[0][1][_cVPOS], dwtMark[0]])] ]
    else:
        if len(dwtData)==1:
            dwtMark.append(dwtData[0][1][_cDPOS])
            dwtData=[ [dwtData[0][0], tuple(dwtMark)] ]
        else:
            # dwtMat = [ [a,(h,v,d)], [a,(h,v,d)] ]
            dwtMark1=[dwtMark[0], dwtData[0][1][_cVPOS], dwtData[0][1][_cDPOS]]
            dwtMark2=[dwtData[1][1][_cHPOS], dwtMark[1], dwtData[1][1][_cDPOS]]

            dwtData=[ [dwtData[0][0], tuple(dwtMark1)], [dwtData[1][0], tuple(dwtMark2)] ]

    newImgData = [pywt.idwt2(x,'db1') for x in dwtData]

    return [newImgData, returnCode]


def _embed(dwtMat,watermark_code):

    dwtMatShape=dwtMat[0].shape

    cH = dwtMat[0].reshape(-1,).tolist()
    cV = dwtMat[1].reshape(-1,).tolist()

    #print "mean CH ", sum(cH)/len(cH), " max: ", max(cH), "min :", min(cH)
    #print "mean CV ", sum(cV)/len(cV), " max: ", max(cV), "min :", min(cV)

    mincV=min(cV)
    maxcH=max(cH)

    _ZEROTHR1 = mincV/1
    _ONETHR1 = maxcH/1

    _ZEROTHR2 = mincV/12
    _ONETHR2 = maxcH/12

    idx=0
    lengthCode=0
    lenMat=len(cH)/2

    for i in watermark_code:
        if idx >= lenMat:
            break

        if i==0:
            while (cV[2*idx] > _ZEROTHR2) and (idx<lenMat-1):
                if cH[2*idx+1] > _ONETHR2:
                    cH[2*idx+1] = 0
                idx+=1
            cH[2*idx+1]=0

            if (cV[2*idx] > _ZEROTHR1):
                cV[2*idx] = _ZEROTHR1

            lengthCode+=1
            idx+=1
        if i==1:
            while (cH[2*idx+1] < _ONETHR2) and (idx<lenMat-1):
                if cV[2*idx] < _ZEROTHR2:
                    cV[2*idx] = 0
                idx+=1
            cV[2*idx]=0

            if (cH[2*idx+1] < _ONETHR1):
                cH[2*idx+1] = _ONETHR1

            lengthCode+=1
            idx+=1

    cH = numpy.array(cH).reshape(dwtMatShape)
    cV = numpy.array(cV).reshape(dwtMatShape)

    return [[cH,cV],lengthCode]


def _extract(dwtMat):

    cH = dwtMat[0].reshape(-1,).tolist()
    cV = dwtMat[1].reshape(-1,).tolist()

    mincV=min(cV)
    maxcH=max(cH)

    _ZEROTHR2 = mincV/4
    _ONETHR2 = maxcH/4

    code=[]
    lenMat=len(cH)/2

    for idx in range(lenMat):
        if (cV[2*idx] < _ZEROTHR2):
            code.append(0)
        elif (cH[2*idx+1] > _ONETHR2):
            code.append(1)

    return code

def _hist(data,window):

    minVal=window[0]#int(round(min(data)-window))
    maxVal=window[1]#int(round(max(data)-window))

    hist=[]

    for i in range(minVal,maxVal-window[2],window[2]):
        hist.append(len([x for x in data if x > i and x <= i+window[2]]))

    return hist