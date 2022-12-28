import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
st.set_page_config(layout="wide")

st.title('Title')
st.header("Header")
st.subheader("Sub Header")

st.write("Hello World!")

@st.cache(allow_output_mutation=True)
def cached_model() :
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset() : 
    df = pd.read_csv("data.csv")
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header("부산소프트웨어마이스터고 챗봇")
st.subheader("안녕하세요 소마고 챗봇입니다.")

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', "")
    submitted = st.form_submit_button("전송")

# 응답 response
if 'generated' not in st.session_state : 
    st.session_state['generated'] = []

# 질문 request
if 'past' not in st.session_state :
    st.session_state['past'] = []

# 예외 처리
if submitted and user_input : 
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x : cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] >= 0.5 : 
        st.session_state.generated.append(answer['챗봇'])
    else :
        st.session_state.generated.append("이 사항을 알고 싶으시면 051-971-2153으로 문의주세요")

for i in range(len(st.session_state['past'])) : 
    message(st.session_state['past'][i], is_user=True, key = str(i) + "_user")
    if len(st.session_state['generated']) > i :     
        message(st.session_state['generated'][i], key = str(i) + "_bot")

st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage](https://school.busanedu.net/bssm-h/main.do) |
    [Instargram](https://www.instargram.com/bssm.hs) |
    [Facebook](https://www.facebook.com/BusanSoftwareMeisterHighschool)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    call: 051-971-2153
    """
) 

tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "문의"])

with tab1:
    st.header("저희 소마고를 소개합니다.")

with tab2:
    st.header("입학 안내")

with tab3:
    st.header("챗봇에게 무엇이든 물어보세요.") 