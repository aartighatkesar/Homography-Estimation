from estimate_homography import *

img_1_pts = [[1518, 181], [2948, 731], [2997, 2046], [1490, 2227]]  #PQSR

img_2_pts =  [[1331, 335], [3014, 621], [3030, 1892], [1309, 2007]]  # PQSR

img_3_pts = [[929, 737], [2799, 390], [2849, 2222], [907, 2079]]  # PQSR

H_ab = calculate_homography(img_1_pts, img_2_pts) # a = H_ab * b

H_bc = calculate_homography(img_2_pts, img_3_pts) # b = H_bc * c

H_ac = np.dot(H_ab, H_bc)

print(H_ac)

print('-------')

print(cv2.findHomography(np.array(img_3_pts), np.array(img_1_pts)))

img_src_path = '/Users/aartighatkesar/Documents/homography_estimation/input_imgs/1.jpg'

img_src = cv2.cvtColor(cv2.imread(img_src_path), cv2.COLOR_BGR2RGB)

img_dst = np.zeros_like(img_src)

mask = np.ones((img_dst.shape[0], img_dst.shape[1]))

out = fit_image_in_target_space(img_src, img_dst, mask, H_ac)

plt.figure()
plt.imshow(out)
plt.axis('off')
plt.savefig('results_task2.jpg')
plt.show()