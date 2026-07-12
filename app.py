import streamlit as st
import pickle

# 1. 깃허브에 올린 진짜 500개 데이터 학습 모델 불러오기
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except Exception as e:
    st.error(f"⚠️ 모델 파일을 불러오는 중 오류가 발생했습니다. 파일이 깃허브에 잘 올라갔는지 확인해주세요: {str(e)}")

# 2. 웹사이트 화면 디자인
st.title("🤖 AI 가짜 뉴스 검출 시스템 (Real AI v2)")
st.markdown("---")
st.write("뉴스 기사를 아래에 입력하면 500개 데이터셋을 학습한 AI 모델이 진짜/가짜 확률을 실시간으로 분석합니다.")

# 유저 인풋 공간
news_text = st.text_area("뉴스 기사 입력", placeholder="여기에 뉴스 기사를 붙여넣으세요...", height=200)

if st.button("AI 모델로 판별하기"):
    if news_text.strip() == "":
        st.warning("⚠️ 뉴스 기사를 입력해주세요!")
    else:
        try:
            # 💡 [진짜 AI 작동] 유저가 입력한 글자를 숫자로 바꾸고 모델에게 넘겨줌
            text_vector = tfidf.transform([news_text])
            prob = model.predict_proba(text_vector)[0]
            
            # 너의 코랩 모델 라벨 기준 (label 0 = 가짜뉴스 / label 1 = 진짜뉴스)
            fake_prob = prob[0] * 100  # 0번 인덱스가 가짜뉴스 확률
            real_prob = prob[1] * 100  # 1번 인덱스가 진짜뉴스 확률
            
            # 결과 화면 출력
            st.subheader("📊 AI 분석 결과")
            st.write(f"- **진짜 뉴스일 확률:** {real_prob:.1f}%")
            st.write(f"- **가짜 뉴스일 확률:** {fake_prob:.1f}%")
            st.markdown("---")
            
            # 확률에 따른 판정
            if fake_prob > 50:
                st.error("🚨 주의: 가짜 뉴스일 가능성이 높습니다! (학습된 왜곡 패턴 감지)")
            else:
                st.success("🍏 확인: 신뢰할 수 있는 진짜 뉴스일 가능성이 높습니다.")
                
        except Exception as e:
            st.error(f"❌ 판별 중 오류 발생: {str(e)}")
