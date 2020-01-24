
# coding: utf-8

# ## Kaggle 입문하기 - 데이터 분석 입문

# ### 학습 내용 
# * 캐글에 대해 이해하기
# * 캐글 데이터 셋을 이용하여 데이터 분석을 이해한다.

#  * URL : https://www.kaggle.com/
#  * Competitions 선택하면 다양한 대회 확인 가능.
#  * 대회 주제 : Bike Sharing Demand
#  * https://www.kaggle.com/c/bike-sharing-demand

# In[7]:


from IPython.display import display, Image


# ## 데이터 다운로드하기
# * 가. https://www.kaggle.com/c/bike-sharing-demand 링크를 선택하여 웹 사이트 접속합니다.
# * 나. Data를 선택합니다.
# * 다. train.csv, test.csv, sampleSubmission.csv를 다운로드 받습니다.
# * 라. 다운로드 받은 csv와 주피터 노트북 또는 py 파일은 동일한 폴더에 위치시킵니다.

# In[36]:


display(Image(filename='img/kaggle/kaggle01.png'))


# * 'Data'를 누르면 데이터 상세 내역이 확인가능합니다.

# In[39]:


display(Image(filename='img/kaggle/kaggle02.png'))


# * 'Data Sources'의 test.csv와 train.csv의 데이터 셋을 다운로드 합니다.

# In[40]:


display(Image(filename='img/kaggle/kaggle03.png'))


# ### Data Fields
# | 필드명 | 설명   |
# |------|:------|
# |   datetime  | hourly date + timestamp   |
# |   season  | 1 = spring, 2 = summer, 3 = fall, 4 = winter  |
# |   holiday  | whether the day is considered a holiday |
# |   workingday  | whether the day is neither a weekend nor holiday |
# |   weather  | <br>1: Clear, Few clouds, Partly cloudy, Partly cloudy<br>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist <br>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds<br>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog <br> |
# |   temp  | temperature in Celsius (온도) |
# |   atemp  | "feels like" temperature in Celsius (체감온도) |
# |   humidity  | relative humidity (습도) |
# |   windspeed  | wind speed (바람속도) |
# |   casual  | number of non-registered user rentals initiated (비가입자 사용유저) |
# |   registered  | number of registered user rentals initiated (가입자 사용유저) |
# |   count  | number of total rentals (전체 렌탈 대수) |

# In[8]:


import pandas as pd


# ### 1-1 데이터 준비하기
# * train 은 학습을 위한 데이터 셋
# * test 은 예측을 위한 데이터 셋
# * ../data/bike : 상위폴더의 (data/bike 폴더 경로), 내 컴퓨터의 데이터 경로 지정.
# * parse_dates = [컬럼명] : 해당 컬럼을 시간형 자료로 불러옴.

# In[21]:


train = pd.read_csv("train.csv", parse_dates=['datetime'])
test = pd.read_csv("test.csv", parse_dates=['datetime'])


# In[22]:


print(train.shape)   # : 행과 열 갯수 확인
print(test.shape)


# In[23]:


train.head()


# In[24]:


train.info()


# ### 입력데이터 선택 

# In[25]:


f_names = ['temp', 'atemp']
X_train = train[f_names]    # 학습용 데이터의 변수 선택 
X_test = test[f_names]      # 테스트 데이터의 변수 선택 


# ### 출력 데이터 선택

# In[26]:


label_name = 'count'        # 렌탈 대수 (종속변수)
y_train = train[label_name] # 렌탈 대수 변수 값 선택


# ### 1-2 모델 만들기 및 제출

# ### 모델 만들기 및 예측 순서
#  * 모델을 생성한다. model = 모델명()
#  * 모델을 학습한다.  model.fit( 입력값, 출력값 )
#  * 모델을 이용하여 예측 model.predict(입력값)

# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


model = LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)   # 예측(새로운 데이터로)


# In[29]:


model.coef_


# In[30]:


model.intercept_


# ### 학습된 모델로 예측 후, 이값으로 제출하기

# In[31]:


pred = model.predict(X_test)   # 예측
sub = pd.read_csv("sampleSubmission.csv")
sub['count'] = pred


# ### 처음 만는 제출용 csv 파일
#  * index=False : csv 파일 행번오 없애기

# In[32]:


# 처음 만는 제출용 csv 파일, 행번호를 없애기
sub.to_csv("firstsubmission.csv", index=False)


# ### 제출하기
# * 캐글 사이트 접속 후, 로그인 
# * 맨 상단에 Search에 Bike Sharing demand로 입력 후, 검색 되는 것 중 하나를 선택
# * 들어간 사이트에서 대회로 접속 후, 
#   * 또는 다음 링크로 접속 : https://www.kaggle.com/c/bike-sharing-demand
# * Late Submission 선택 후, 제출 영역에 csv 파일을 마우스 드래그하여 올려 제출
# * 제출 후, 아래 'Make Submission' 을 버튼을 선택하면 제출 결과가 약간 후 보임.

# In[33]:


display(Image(filename='img/kaggle/bike01.png'))


# In[34]:


## 업로드가 완료된 후, 아래 버튼 선택
display(Image(filename='img/kaggle/bike01.png'))

