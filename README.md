# Project Introduction

![image](https://github.com/ryeonbeenkang/mulcam_1st_team_project/assets/47935123/d21abc00-1f66-45df-aa4f-2fbafe73852e)


해당 프로젝트는 멀티캠퍼스에서 Decision Tree계열의 머신러닝 알고리즘을 학습한 후, 복습을 목적으로 진행된 프로젝트이며, 'Kaggle'의 'Mushroom Classification'이라는 데이터 셋 속의 다양한 버섯들의 속성들을 활용 & 학습하여 식용여부가 결정되지 않은 버섯들들의 'class'(식용 가능 여부를 나타내는 지표)를 머신러닝 기법으로 학습 하여 식용여부를 판단하여보는 프로젝트 였다. 2023년 11월 28일 부터 21월 1일까지 4일간 매일 3시간씩 진행 되었으며, 결과물은 별도의 업로드 사이트에 업로드 하여 그 결괏값을 측정하는 방식으로 진행 되었다. 

(출처: https://www.kaggle.com/datasets/uciml/mushroom-classification)



# Data sets
## 1. 데이터 특성 파악(EDA분석)
### 컬럼들의 파악
총 컬럼수 : 24개(단, class, mushroom_id는 논외)

(1) cap-shape(Feature / Categorical) = 버섯 갓 모양(bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s)

(2) cap-surface(Feature / Categorical) = 버섯 갓 표면(fibrous=f,grooves=g,scaly=y,smooth=s)

(3) cap-color(Feature / Categorical) = 버섯 갓 표면색(brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y)

(4) bruises(Feature / Categorical) = 버섯의 멍(bruises=t,no=f)

(5) odor(Feature / Categorical) = 냄새(almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s)

(6) gill-attachment (Feature / Categorical) = 주름살 부착(attached=a,descending=d,free=f,notched=n)

(7) gill-spacing(Feature / Categorical) = 주름살 간격(close=c,crowded=w,distant=d)

(8) gill-size(Feature / Categorical) = 주름살 사이즈(broad=b,narrow=n)

(9) gill-color(Feature / Categorical) = 주름살 색(black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y)

(10) stalk-shape(Feature / Categorical) = 대 모양(enlarging=e,tapering=t)

(11) stalk-root(Feature / Categorical) = 대 뿌리(bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?)

(12) stalk-surface-above-ring(Feature / Categorical) = 턱받이 위의 대 표면(fibrous=f,scaly=y,silky=k,smooth=s)

(13) stalk-surface-below-ring(Feature / Categorical) = 턱받이 아래의 대 표면(fibrous=f,scaly=y,silky=k,smooth=s)

(14) stalk-color-above-ring(Feature / Categorical) = 턱받이 위 대 색깔(brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y)

(15) stalk-color-below-ring(Feature / Categorical) = 턱받이 아래 대 색깔(brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y)

(16) veil-type(Feature / Binary) = 베일(: 신부 면사포 같이 생긴것) 타입(partial=p,universal=u)

(17) veil-color(Feature / Categorical) = 베일 색깔(brown=n,orange=o,white=w,yellow=y)

(18) ring-number(Feature / Categorical) = 턱받이 수(none=n,one=o,two=t)

(19) ring-type(Feature / Categorical) = 턱받이 타입(cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z)

(20) spore-print-color(Feature / Categorical) = 버섯 기공(피부의 구멍과 같은) 색깔(black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y)

(21) population(Feature / Categorical) = 버섯이 얼마나 흔한지(개체수)(abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y)

(22) habitat(Feature / Categorical) = 서식지(grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d)

(23) class = 생존여부((edible=e, poisonous=p))

(24) mushroom_id = 인덱스
 
 - train셋 / test셋: 6500을 기준으로 분리


# 가설
  [파생변수 생성을 위한 가설]
   1) 색이 화려한 버섯은 독버섯이다 - cap-color을 dummie화 시킴
      
   2) 대에 띠가 없는 버섯은 독버섯이다 - ring-number, ring-type을 dummies화 시킴
      
   3) 세로 결이 없고, 세로로 잘 찢어지지 않는 버섯이 독버섯이다 - gill-attachment를 dummies화 시킴


# 전처리
 - 영향이 적을 것으로 예상되는 컬럼들을 제거:
   ```
   df = df.drop(["stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",\
   "stalk-color-above-ring","stalk-color-below-ring","gill-spacing","gill-color","odor","cap-shape",\
   "cap-surface","bruises","veil-color","spore-print-color","population","habitat"], axis=1)

   ```
 - 인코딩 & 수치화:
   (1) Label Encodingg할 컬럼들(0,1로 표현) - (1)gill-size, (2)stalk-shape, (3)veil-type, (4)ring-number

  
   (2) One Hot Encoding할 것들(3개 이상의 속성값을 가진 순서가 없는 컬럼 dummy화) - (1)gill-attachment, (2)gill-spacing, (3)gill-color, (4)stalk-root, \
      (5)stalk-surface-above-ring, (6)stalk-surface-above-ring, (7)stalk-color-above-ring, (8)stalk-color-below-ring, \
      (9)veil-color, (10)ring-type, (11)spore-print-color, (12)habitat

      
   (3) class 컬럼을 'e'=0 / 'p'=1로 수치화

  
 - 결측치: 결과 도출을 위한 test 데이터 프래임의 class컬럼이 비어 있는것을 제외하곤, 결측치는 존재 하지 않았다. 


# 머신러닝 기법
 - (1) DecisionTree, (2)RandomForest, (3)BoostingTree(lightGBM) - 순서대로 적용해 보고 결괏값을 비교


# 보조기능
(1) Label Encoding

(2) One Hot Encoding

(3) Entropy

(4) K-fold validation

(5) Stratified K-fold validation

(6) Hypter Parameter Optimization

(7) Graphviz

(8) Feature_importances_



# 결과
Decision Tree계열 알고리즘을 사용하여 분류 모델 학습 시키고 검증 + 테스트 셋에 대하여 인퍼런스 한 뒤, 결과 제출



# 인사이트
1. 아무관련이 없을 것 같았던 버섯의 띠 보유 현황이 독성판별에 영향을 줬다는 점
   
2. 생각보다 다양한 형태를 지닌 독버섯이 존재 했다점
   
3. 사람들이 경험적으로 '색깔이 이런 버섯은 독버셧이야' 하는 말이 실제 머신러닝 과정을 거쳐 증명된 점



# 어려웠던점 & 한계점
1) 버섯에 대한 무지. 많은 시간을 버섯에 대한 조사에 할애
   
2) 총 22가지의 컬럼이 존재, 각 특성의 파악에 장시간을 소요
   
3) 대부분의 컬럼 데이터들이 카테고리형 데이터에 순서가 없는 형태였으므로, 많은 dummy 화가 필요
   
4) 많은 컬럼수로 인하여 모든 컬럼들을 전부 평가해 볼 수는 없었던 점이 한계점
   
5) DecisionTree이 max_depth를 어떻게 적절히 설정할 수 있는지 여러 반복을 거쳐 봤어야 했던점
    
6) feature_importances_로 추려낸 중요한 feature들로 모델 학습을 시켜 보아도 그다지 성능의 향상이 크게 보이지 않았던점
    
7) gini나 log_loss에 대한 이해부족으로 인한 대부분의 학습이 entropy가 criterion으로 설정되어 학습된점. 

