__author__ = 'liuc'

import Image, numpy

import mineWatermark
import ssim


def main():
    #watermark('lena.png','lena_out.png','Copyright by Luca','#ffffff','chintzy.ttf',60,(0.5,0.98),0.7)
    #testdwt('lena.png')

    img1 = Image.open('lena.png')#.convert('L')

    [img2, lengthCode] = mineWatermark.insert(img1,[1,0]*100000,3)

    print "Code inserted = " , lengthCode

    img2.save('out.jpg','JPEG',quality=90)

    img2 = Image.open('out.jpg')

    img1G = numpy.array(img1.convert('L').getdata()).reshape(img1.size)
    img2G = numpy.array(img2.convert('L').getdata()).reshape(img2.size)

    res = ssim.ssim(img1G,img2G)
    print res

    code=mineWatermark.extract(img2,3)

    print len(code), " ",code

if __name__ == '__main__':
    main()