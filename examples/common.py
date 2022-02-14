import cv2 as cv
import numpy as np

# See https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

ordinalltx = lambda n: r"%d$^{%s}$" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def deskew(img, raster_sz = 20):
    affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*raster_sz*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(raster_sz, raster_sz),flags=affine_flags)
    return img

def hog(img, bins_n = 16):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bins_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bins_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

# def img2VectorsWithInfluence(img, imgPro, imgCon, raster_sz = 20, ret_img_sz = 128):
#     gx = cv.Sobel(img, cv.CV_32F, 1, 0)
#     gy = cv.Sobel(img, cv.CV_32F, 0, 1)

#     gxPro = cv.Sobel(imgPro, cv.CV_32F, 1, 0)
#     gyPro = cv.Sobel(imgPro, cv.CV_32F, 0, 1)

#     gxCon = cv.Sobel(imgCon, cv.CV_32F, 1, 0)
#     gyCon = cv.Sobel(imgCon, cv.CV_32F, 0, 1)

#     # Create a black image
#     imgGradSize = ret_img_sz

#     imgVectors = np.zeros((imgGradSize,imgGradSize,3), np.uint8)
#     scaleFactor = imgGradSize/raster_sz
    
#     maxX = np.max(gx)
#     maxY = np.max(gy)
#     maxXY = max(maxX,maxY)

#     maxXPro = np.max(gxPro)
#     maxYPro = np.max(gyPro)
#     maxXYPro = max(maxXPro,maxYPro)

#     maxXCon = np.max(gxCon)
#     maxYCon = np.max(gyCon)
#     maxXYCon = max(maxXCon,maxYCon)
#     i = 0
    
#     while i<raster_sz:
#         j = 0
#         while j<raster_sz:
#             pt1 = (np.int32(i*scaleFactor),np.int32(j*scaleFactor))

#             pt2 = (np.int32((i+gx[j][i]/maxXY)*scaleFactor), np.int32((j+gy[j][i]/maxXY)*scaleFactor))
#             pt2Pro = (np.int32((i+gxPro[j][i]/maxXYPro)*scaleFactor), np.int32((j+gyPro[j][i]/maxXYPro)*scaleFactor))
#             pt2Con = (np.int32((i+gxCon[j][i]/maxXYCon)*scaleFactor), np.int32((j+gyCon[j][i]/maxXYCon)*scaleFactor))

#             r = 0
#             g = 0
#             b = 0
#             if (gx[j][i] + gy[j][i] > 0):
#                 if (gxPro[j][i] + gyPro[j][i] > 0):
#                     g = 255
#                     pt2Pro = (np.int32((i+(gx[j][i]/maxXY)*(gxPro[j][i]/maxXYPro))*scaleFactor), np.int32((j+(gy[j][i]/maxXY)*(gyPro[j][i]/maxXYPro))*scaleFactor))
#                     imgVectors = cv.arrowedLine(imgVectors, pt1, pt2Pro, (r,g,b) ) 
#                 elif (gxCon[j][i] + gyCon[j][i] > 0):
#                     r = 255
#                     pt2Con = (np.int32((i+(gx[j][i]/maxXY)*(gxCon[j][i]/maxXYCon))*scaleFactor), np.int32((j+(gy[j][i]/maxXY)*(gyCon[j][i]/maxXYCon))*scaleFactor))
#                     imgVectors = cv.arrowedLine(imgVectors, pt1, pt2Con, (r,g,b) ) 
#                 else:
#                     r = 255
#                     g = 255
#                     b = 255
#                     imgVectors = cv.arrowedLine(imgVectors, pt1, pt1, (r,g,b) ) 
#             else:
#                 if (gxPro[j][i] + gyPro[j][i] > 0):
#                     r = 0
#                     g = 255
#                     b = 255
#                     imgVectors = cv.arrowedLine(imgVectors, pt1, pt2Pro, (r,g,b) ) 
#             j = j + 1
#         i = i + 1


#     return imgVectors

# def img2Gradients(img, raster_sz = 20, ret_img_sz = 128):
#     gx = cv.Sobel(img, cv.CV_32F, 1, 0)
#     gy = cv.Sobel(img, cv.CV_32F, 0, 1)

#     # Create a black image
#     imgGradSize = ret_img_sz

#     imgVectors = np.zeros((imgGradSize,imgGradSize,3), np.uint8)
#     scaleFactor = imgGradSize/raster_sz
    
#     maxX = np.max(gx)
#     maxY = np.max(gy)
#     maxXY = max(maxX,maxY)

#     i = 0
    
#     while i<raster_sz:
#         j = 0
#         while j<raster_sz:
#             pt1 = (np.int32(i*scaleFactor),np.int32(j*scaleFactor))
#             pt2 = (np.int32((i+gx[j][i]/maxXY)*scaleFactor), np.int32((j+gy[j][i]/maxXY)*scaleFactor))
#             if (gx[j][i] + gy[j][i] > 0):
#                 r = 255
#                 g = 255
#                 b = 255
#                 imgVectors = cv.arrowedLine(imgVectors, pt1, pt2, (r,g,b) ) 
#             j = j + 1
#         i = i + 1

#     return imgVectors


def get_imgWithInfluence(img, raster_sz = 20, ret_img_sz = 128, fg_color=(0,0,0), bg_color=(255,255,255)):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)

    # Create a black image
    imgGradSize = ret_img_sz

    imgVectors = np.zeros((imgGradSize,imgGradSize,3), np.uint8)
    imgVectors = cv.rectangle(imgVectors, (0,0), (imgGradSize,imgGradSize), bg_color, -1 ) 
    scaleFactor = imgGradSize/raster_sz
    
    maxX = np.max(np.abs(gx))
    maxY = np.max(np.abs(gy))
    maxXY = max(maxX,maxY)

    i = 0
    
    while i<raster_sz:
        j = 0
        while j<raster_sz:
            pt1 = (np.int32(i*scaleFactor),np.int32(j*scaleFactor))
            pt2 = (np.int32((i+gx[j][i]/maxXY)*scaleFactor), np.int32((j+gy[j][i]/maxXY)*scaleFactor))
           
            r = 0
            g = 0
            b = 0
            if (gx[j][i] + gy[j][i] > 0):
                r = fg_color[0]
                g = fg_color[1]
                b = fg_color[2]
                imgVectors = cv.arrowedLine(imgVectors, pt1, pt2, (r,g,b) ) 
               
            j = j + 1
        i = i + 1

    return imgVectors


def get_influenceMap(img,  imgV, positiveMISV, negativeMISV, raster_sz = 20, ret_img_sz = 128,  bins_n = 16,  bg_color=(255,255,255)):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    
    _ , ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bins_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    half_raster_sz = np.int32(raster_sz/2)
    proInfluence = imgV * positiveMISV
    conInfluence = imgV * negativeMISV

    maxProInfluence = np.max(np.abs(proInfluence))
    maxConInfluence = np.max(np.abs(conInfluence))

    proInfluence = proInfluence/max(maxProInfluence, 1)
    conInfluence = conInfluence/max(maxConInfluence, 1)
   
    # Create a black image
    imgGradSize = ret_img_sz

    imgVectors = np.zeros((imgGradSize,imgGradSize,3), np.uint8)
    imgVectors = cv.rectangle(imgVectors, (0,0), (imgGradSize,imgGradSize), bg_color, -1 ) 
    scaleFactor = imgGradSize/raster_sz
    
    maxX = np.max(np.abs(gx))
    maxY = np.max(np.abs(gy))
    maxXY = max(maxX,maxY)

    offsetInfluence = 0

    for i in range(raster_sz):

        for j in range(raster_sz):
            offsetInfluence = int((i+j)/half_raster_sz)*bins_n
            pt1 = (np.int32((i)*scaleFactor),np.int32((j)*scaleFactor))
            pt2 = (np.int32(((i)+gx[j][i]/maxXY)*scaleFactor), np.int32((j+gy[j][i]/maxXY)*scaleFactor))
            r = 0
            g = 0
            b = 0
            idx_angle = bins[j][i]
            if (gx[j][i]) + (gy[j][i]) > 0:
                if proInfluence[idx_angle+offsetInfluence] +  conInfluence[idx_angle+offsetInfluence] != 0:
                    if  proInfluence[idx_angle+offsetInfluence] > conInfluence[idx_angle+offsetInfluence]:
                        # (32,133,64)
                        r =  32
                        g =  100 + int(33 * proInfluence[idx_angle+offsetInfluence])
                        b =  64
                    else:
                        # (0, 113, 188)
                        r =  0
                        g =  113 
                        b =  150 + int(38 * proInfluence[idx_angle+offsetInfluence])
                        
                else:
                    r = 0   
                    g = 0
                    b = 0
                imgVectors = cv.arrowedLine(imgVectors, pt1, pt2, (r,g,b) ) 
               
    return imgVectors