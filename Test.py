import numpy as np
import cv2
import os
from os import path
from matplotlib import pyplot as plt
# os.chdir("ChessBoard") 
# img = cv2.imread('left01.jpg',cv2.IMREAD_GRAYSCALE)
# sift = cv2.xfeatures2d.SIFT_create()
# keypoints_sift, descriptors = sift.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img, keypoints_sift, None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

MIN_MATCH_COUNT = 10


object_name = ['champion_copper_plus_spark_plug', 'cheezit_big_original','crayola_64_ct',
'dr_browns_bottle_brush','elmers_washable_no_run_school_glue','expo_dry_erase_board_eraser',
'feline_greenies_dental_treats','first_years_take_and_toss_straw_cup','genuine_joe_plastic_stir_sticks',
'highland_6539_self_stick_notes','kong_air_dog_squeakair_tennis_ball','kong_duck_dog_toy',
'kyjen_squeakin_eggs_plush_puppies','laugh_out_loud_joke_book','mark_twain_huckleberry_finn','mommys_helper_outlet_plugs',
'munchkin_white_hot_duck_bath_toy','oreo_mega_stuf','paper_mate_12_count_mirado_black_warrior','rolodex_jumbo_pencil_cup',
'safety_works_safety_glasses','sharpie_accent_tank_style_highlighters','stanley_66_052',]

image_type = ['depth','image','mask','']
shelf_name = ['A','B','C','D','E','F','G','H','I','L']
num_one = ['1','2','3']
num_two = ['0','1','2','3']
num_three = ['0','1','2','3']


'May use this nested for loop below if I decide to loop through all of the items'
# for obj in object_name:
#     for shelf in shelf_name:
#         for i in range(3):
#             for j in range(4):
#                 for k in range (4):
#                     for image in image_type:
#                         item = (f'{obj}-{image}-{shelf}-{i}-{j}-{k}')
#                         if item = 




substring1 = 'image'
substring2 = 'mask'
for obj in object_name:
    for shelf in shelf_name:
        for image in image_type:
            item = f'{obj}-{image}-{shelf}'
            if path.exists(f'{obj}-{image}-{shelf}-1-1-0.png') == False:
                continue

            if image == 'depth':                # Reading in any depth images found
                item_name = f'{obj}-{image}-{shelf}-1-1-0.png'
                depth = cv2.imread(item_name,-1)

            if image == 'image':                # Reading in any rgb images found
                item_name = f'{obj}-{image}-{shelf}-1-1-0.png'
                rgbImage = cv2.imread(item_name)

            if image == 'mask':                 # Reading in any mask images found

                'Applying the mask to the RGB and depth images'

                mask = cv2.imread(f'{obj}-{image}-{shelf}-1-1-0.png')

                depthMask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                depthMask = depthMask.astype(np.uint16)
                depthFinal = depth

                depthFinal[np.where(depthMask != 255)] = 0 #Couldn't do bitwise AND because 8 bit grayscale vs. 16 bit depth
                item3 = cv2.bitwise_and(rgbImage, mask)
                
                'Harris corner detector'

                grayitem3 = cv2.cvtColor(item3,cv2.COLOR_BGR2GRAY)
                gray = np.float32(grayitem3)
                dst = cv2.cornerHarris(gray,2,3,0.04)
                cv2.imshow('sdf',dst)
                cv2.waitKey()
                'Found corners, found depth should be less than 950ish or else it is at the back wall. How to account for edge mismatches or other mismatches that are not far away, depth-wise? How to combine known corners with known depth to find where the item likely is? Then SIFT, etc.'
                
                item3[dst>0.01*dst.max()]=[255,0,0]
                print(np.count_nonzero(dst), dst)
                print(item3.size)
                x,y = np.where(dst>0.01*dst.max())
                for corner in x,y:
                    
                # for corner in x,y:
                #     print(corner)
                'Refining corner detector'
                # ret, dst = cv2.threshold(item3,0.01*dst.max(),255,0)
                # dst = np.uint8(dst)
                # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

                # res = np.hstack((centrimg=[0,0,255]
                # img[res[:,3],res[:,2]] = [0,255,0]


                # cv2.imshow('dst',depthFinal)
                # cv2.waitKey(0)

                cv2.imshow('dst',item3)
                cv2.waitKey(0)
                # gray = cv2.cvtColor(item3,cv2.COLOR_BGR2GRAY)
                # thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
                # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # cnt = contours[0]
                # x,y,w,h = cv2.boundingRect(cnt)
                print(f'{obj}-{image}-{shelf}-1-1-0.png', f'{obj}.png')
                img1 = cv2.imread(f'../tarball/{obj}.png')          # queryImage
                grayimg1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
                # Initiate SIFT detector
                sift = cv2.xfeatures2d.SIFT_create()

                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(img1,None)
                kp2, des2 = sift.detectAndCompute(item3,None)

                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks = 100)


                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(des1,des2,k=2)

                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                if len(good)>MIN_MATCH_COUNT:
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    if M is None:
                        continue
                    matchesMask = mask.ravel().tolist()

                    h,w = img1.shape[:-1]
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)

                    item3 = cv2.polylines(item3,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                else:
                    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
                    matchesMask = None



                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

                img_match = cv2.drawMatches(img1,kp1,item3,kp2,good,None,**draw_params)
            
                # cv2.imshow("Match",img_match)
                # cv2.waitKey(0)
