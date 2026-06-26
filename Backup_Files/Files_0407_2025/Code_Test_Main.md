### Code of Testing in Main Func

---

- ```C++
  int main(void) {
  	Mat oriImg[14];
  	Mat resImg[14];
  	char inputPathName_ori[14][256];
  	char outputPathName_res[14][256];
  
  	for (int idxImg = 0; idxImg < 14; idxImg++) {
  		if (idxImg < 9) {
  			sprintf_s(inputPathName_ori[idxImg], "./imgs_0407_2025_v0/input/oriImg_0%d.png", idxImg + 1);
  		}
  		else {
  			sprintf_s(inputPathName_ori[idxImg], "./imgs_0407_2025_v0/input/oriImg_%d.png", idxImg + 1);
  		}
  		oriImg[idxImg] = imread(inputPathName_ori[idxImg], 0);
  	}
  
  	for (int idxImg = 0; idxImg < 14; idxImg++) {
  		imgSingleProcess(oriImg[idxImg], resImg[idxImg], info_val_dv);
  	}
  
  	for (int idxImg = 0; idxImg < 14; idxImg++) {
  		if (idxImg < 9) {
  			sprintf_s(outputPathName_res[idxImg], "./imgs_0407_2025_v0/output/Version_01/resImg_0%d.png", idxImg + 1);
  		}
  		else {
  			sprintf_s(outputPathName_res[idxImg], "./imgs_0407_2025_v0/output/Version_01/resImg_%d.png", idxImg + 1);
  		}
  		imwrite(outputPathName_res[idxImg], resImg[idxImg]);
  	}
  
  	return 0;
  }
  ```

- 