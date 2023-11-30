# Project Introduction

![image](https://github.com/ryeonbeenkang/mulcam_1st_team_project/assets/47935123/d21abc00-1f66-45df-aa4f-2fbafe73852e)


# Data sets
## 1. 데이터 특성 파악(EDA분석)
### 컬럼들의 파악
총 컬럼수 : 24개

(1) cap-shape(Feature / Categorical) = 버섯 갓 모양	

(2) cap-surface(Feature / Categorical) = 버섯 갓 표면

(3) cap-surface(Feature / Categorical) = 버섯 갓 표면

(4) bruises(Feature / Categorical) = 버섯의 멍

(5) odor(Feature / Categorical) = 냄새

(6) gill-attachment (Feature / Categorical) = 주름살 부착

(7) gill-spacing(Feature / Categorical) = 주름살 간격

(8) gill-size(Feature / Categorical) = 주름살 사이즈

(9) gill-color(Feature / Categorical) = 주름살 색

(10) stalk-shape(Feature / Categorical) = 대 모양

(11) stalk-root(Feature / Categorical) = 대 뿌리	

(12) stalk-surface-above-ring(Feature / Categorical) = 턱받이 위의 대 표면	

(13) stalk-surface-below-ring(Feature / Categorical) = 턱받이 아래의 대 표면

(14) stalk-color-above-ring(Feature / Categorical) = 턱받이 위 대 색깔	

(15) stalk-color-below-ring(Feature / Categorical) = 턱받이 아래 대 색깔

(16) veil-type(Feature / Binary) = 베일(: 신부 면사포 같이 생긴것) 타입

(17) veil-color(Feature / Categorical) = 베일 색깔	

(18) ring-number(Feature / Categorical) = 턱받이 수

(19) ring-type(Feature / Categorical) = 턱받이 타입

(20) spore-print-color(Feature / Categorical) = 버섯 기공(피부의 구멍과 같은) 색깔	

(21) population(Feature / Categorical) = 버섯이 얼마나 흔한지(개체수)

(22) habitat(Feature / Categorical) = 서식지

(23) class = 생존여부

(24) mushroom_id = 인덱스
 
 - train
 - test


# 전처리
 - 영향이 적을 것으로 예상되는 컬럼들을 제거
   ```
   df = df.drop(["stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","gill-spacing","gill-color","odor","cap-shape","cap-surface","bruises","veil-color","spore-print-color","population","habitat"], axis=1)

   ```
 - 인코딩
   1) Label Encodingg할 컬럼들(0,1로 표현) - (1)gill-size, (2)stalk-shape, (3)veil-type, (4)ring-number
   2) One Hot Encoding할 것들(순서 없음, dummy화) - (1)gill-attachment, (2)gill-spacing, (3)gill-color, (4)stalk-root, (5)stalk-surface-above-ring, (6)stalk-surface-above-ring, (7)stalk-color-above-ring, (8)stalk-color-below-ring, (9)veil-color, (10)ring-type, (11)spore-print-color, (12)habitat

 
 - 수치화
   1) class 컬럼을 'e'=0 / 'p'=1로 수치화
  
 - 결측치: 결과 도출을 위한 test 데이터 프래임의 class컬럼이 비어 있는것을 제외하곤, 결측치는 존재 하지 않았다. 


# 가설
  [파생변수 생성을 위한 가설]
   1. 색이 화려한 버섯은 독버섯이다 - cap-color을 dummie화 시킴
   2. 대에 띠가 없는 버섯은 독버섯이다 - ring-number, ring-type을 dummies화 시킴
   3. 세로 결이 없고, 세로로 잘 찢어지지 않는 버섯이 독버섯이다 - gill-attachment를 dummies화 시킴



# 머신러닝
 - 사용 모델: DecisionTree, RandomForest, BoostingTree(lightGBM)



# 결과
## Decision Tree계열 알고리즘을 사용하여 분류 모델 학습 시키고 검증


## 테스트 셋에 대하여 인퍼런스 한 뒤, 결과 제출하기



# 인사이트


# 한계점 & 어려웠던점
1) 어려 웠던 점: 버섯에 대한 무지. 많은 시간을 버섯에 대한 조사에 할애

